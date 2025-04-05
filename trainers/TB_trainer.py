import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from supar import Parser
from trainers.base_trainer import SimpleTrainer

import utils.lcs as lcs
import utils.tlite as tlite
from utils.model_loader import print_gpu_memory

sys.path.append("..")


class TB_trainer(SimpleTrainer):

    def __init__(self, maxiter, patience, train_seed, seed, num_compose, num_candidates, backbone):
        super(TB_trainer, self).__init__(maxiter, patience, train_seed, seed, num_compose, num_candidates, backbone)
        self.patience_counter = 1
        self.W_candidates = []
        self.W_scores = []
        self.original_candidate = None
        self.original_score = None
        self.result_candidate = None
        self.result_score = None
        self.parser = Parser.load("crf-con-en")
        self.para_tokenizer = None
        self.para_model = None
        self.tabu_table = []
        self.state = {}

    def get_state(self, current_iteration, delete_tracker):
        self.state = {
            "np_random_state": np.random.get_state(),
            "random_state": random.getstate(),
            "current_iteration": current_iteration,
            "W_candidates": self.W_candidates,
            "W_scores": self.W_scores,
            "result_candidate": self.result_candidate,
            "result_score": self.result_score,
            "patience_counter": self.patience_counter,
            "delete_tracker": delete_tracker,
        }

    def set_state(self):
        current_iteration = self.state["current_iteration"]
        delete_tracker = self.state["delete_tracker"]
        self.W_candidates = self.state["W_candidates"]
        self.W_scores = self.state["W_scores"]
        self.result_candidate = self.state["result_candidate"]
        self.result_score = self.state["result_score"]
        self.patience_counter = self.state["patience_counter"]
        np.random.set_state(self.state["np_random_state"])
        random.setstate(self.state["random_state"])
        return current_iteration, delete_tracker

    def tournament_selection(self):
        S_candidates = []
        S_scroes = []
        for k in range(self.num_tournaments):
            parent = np.random.randint(0, len(self.W_candidates))  # parent, score_parent <-- Random(W)
            S_candidates.append(self.W_candidates[parent])  # S_candidate = S_candidate + parent
            S_scroes.append(self.W_scores[parent])  # S_score = S_score + score_parent
        base_idx = np.argmax(S_scroes)  # base_idx = \arg max_{idx \in S} S_score
        base_candidate = S_candidates[base_idx]  # base <-- S_candidates(base_idx)
        base_score = S_scroes[base_idx]  # base_score <-- S_candidates(base_idx)

        return base_candidate, base_score

    def containenglish(self, str0):
        return bool(re.search("[a-z A-Z]", str0))

    def mutated_tabu(self, base_candidate, phrase_lookup, use_add, delete_tracker, edit_operations, args):

        deleted = {}
        added = {}

        if base_candidate == self.original_candidate:
            for p in phrase_lookup.values():
                print(p)
        if use_add:
            if len(delete_tracker):
                if "add" not in edit_operations:
                    edit_operations.append("add")
            else:
                if "add" in edit_operations:
                    edit_operations.remove("add")

        empty = True
        while empty:
            if self.num_compose == 1:
                edits = np.random.choice(edit_operations, self.num_candidates)
            else:
                edits = []
                for n in range(self.num_candidates):
                    edits.append(np.random.choice(edit_operations, self.num_compose))
            print(edits)

            # generate candidates
            candidates = []
            for edit in edits:
                if isinstance(edit, str):
                    candidate, indices = self.perform_edit(edit, base_candidate, phrase_lookup, delete_tracker)
                    empty = not self.containenglish(candidate)
                    if not empty:
                        print(candidate)
                        candidates.append(candidate)
                        if edit == "del":
                            deleted[candidate] = [phrase_lookup[indices[0]]]
                        if edit == "add":
                            if len(indices):
                                added[candidate] = indices
                    else:
                        print("""Note: The mutated candidate is an empty string, and it is deleted.""")
                else:
                    old_candidate = base_candidate
                    composed_deletes = []
                    composed_adds = []
                    for op in edit:
                        phrase_lookup = self.get_phrase_lookup(old_candidate, args)
                        new_candidate, indices = self.perform_edit(op, old_candidate, phrase_lookup, delete_tracker)
                        empty = not self.containenglish(new_candidate)
                        if not empty:
                            print(new_candidate)
                            if op == "del":
                                composed_deletes.append(phrase_lookup[indices[0]])
                            if op == "add":
                                if len(indices):
                                    composed_adds.append(indices[0])
                            old_candidate = new_candidate
                        else:
                            break

                    if not empty:
                        candidates.append(new_candidate)
                        if "del" in edit:
                            deleted[new_candidate] = composed_deletes
                        if "add" in edit and len(composed_adds) > 0:
                            added[new_candidate] = composed_adds
        scores = []
        tabu_candidates = []
        for c, candidate in enumerate(candidates):
            tabu_res = self.tabu(candidate, mode="match")
            if tabu_res == 0:
                tabu_candidates.append(candidate)
                scores.append(self.score(candidate, args=args))
                print(scores[-1])
                print_gpu_memory()

        return tabu_candidates, scores, deleted, added

    def tabu(self, candidate, mode="match", temp=0.5, thre=0.5):
        if mode == "match":
            if candidate in self.tabu_table:
                if temp >= np.random.random():
                    return 0
                else:
                    return 1
            else:
                return 0
        else:
            lc = []  # list of the longest common subseqences / subsrtings
            for item in self.tabu_table:
                if mode == "lcsq":  # similarity by longest common subseqences
                    candidate = " " + candidate
                    item = " " + item
                    lc.append(lcs.hunt_szymanski(candidate, item))
                elif mode == "lcss":  # similarity by longest common subsrtings
                    lc.append(lcs.lcs_ukkonen(candidate, item))

            avg_len = len("".join(self.tabu_table)) / len(
                self.tabu_table
            )  # average length of the candidates in tabu table
            lcsq_avg_len = len("".join(lc)) / len(lc)  # average length of the longest common subseqences / subsrtings
            similarity = lcsq_avg_len / avg_len

            if similarity >= thre:
                if temp >= np.random.random():
                    return 0
                else:
                    return 1
            else:
                return 0

    def train(self, instruction, chosen_task_name, args):

        N_tabu = 5

        meta_path = os.path.join(args.meta_dir, args.meta_name)
        meta_file = open(meta_path, "w+")
        edit_operations = args.edits
        use_add = "add" in edit_operations

        if "sub" in edit_operations:
            self.if_sub(edit_operations)

        self.init_population(instruction, args)
        self.tabu_table.append(self.original_candidate)

        meta_file.write("Original Candidate:\t " + self.original_candidate + "\n")
        meta_file.write("Original Score:\t " + str(self.original_score) + "\n")
        meta_file.write("\n")
        current_iteration = 0
        delete_tracker = []

        if len(args.resume):
            print("Resuming the searching from checkpoints...")
            self.load(args.resume)
            current_iteration, delete_tracker = self.set_state()

        while current_iteration < self.maxiter:
            current_iteration += 1
            # Base_candidate after battled in the tournament
            base_candidate = self.result_candidate
            base_score = self.result_score

            meta_file.write("Base Candidate:\t " + base_candidate + "\n")
            meta_file.write("Base Score:\t " + str(base_score) + "\n")

            # when the error (caused by parser) occurs, delete the corresponding candidate and its score from the
            # population W
            try:
                phrase_lookup = self.get_phrase_lookup(base_candidate, args)
            except AttributeError:
                self.W_scores.remove(self.W_scores[self.W_candidates.index(base_candidate)])
                self.W_candidates.remove(base_candidate)
                meta_file.write("AttributeError occurs (parser) and skip this iteration" + "\n")
                print("AttributeError occurs (parser) and skip this iteration")
                self.result_score = self.W_scores[-1]
                self.result_candidate = self.W_candidates[-1]
                continue

            candidates, scores, deleted, added = self.mutated_tabu(
                base_candidate, phrase_lookup, use_add, delete_tracker, edit_operations, args
            )

            if not len(candidates) == 0:
                best_score, best_candidate = self.choose_best(candidates, scores)
                self.tabu_table.append(best_candidate)
            else:
                continue

            if len(self.tabu_table) > N_tabu:
                self.tabu_table.pop(0)
            use_simulated_anneal = args.simulated_anneal

            if use_simulated_anneal:
                add_best_or_not = self.update_result_add(
                    best_score, best_candidate, use_simulated_anneal, current_iteration
                )
            else:
                add_best_or_not = self.update_result_add(best_score, best_candidate)

            if add_best_or_not:
                self.W_candidates.append(best_candidate)
                self.W_scores.append(best_score)

                if self.result_candidate in added.keys():
                    print("Notice! Prev tracker: ", delete_tracker)
                    for chunk in added[self.result_candidate]:
                        try:
                            delete_tracker.remove(chunk)
                        except:
                            pass
                    print("Notice! New tracker: ", delete_tracker)

                if self.result_candidate in deleted.keys():
                    delete_tracker.extend(deleted[self.result_candidate])

                # self.result_candidate = self.detokenize(self.word_tokenize(self.result_candidate))

            if current_iteration % args.checkpoint_freq == 0:
                self.get_state(current_iteration, delete_tracker)
                ckpt_dir = Path(args.output_dir) / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                filename = "task{}_step{}.pickle".format(args.task_idx, current_iteration - 1)
                ckpt_path = ckpt_dir / filename
                self.save(ckpt_path)

            if args.backbone == "tlite":
                count = tlite.complete_tlite.count

            # if count >= args.budget:
            #     print('Ran out of budget')
            #     break

            if self.patience_counter > args.patience:
                print("Ran out of patience")
                meta_file.write("Ran out of patience \n")
                break
            elif count >= args.budget:
                print("Ran out of budget")
                break
            else:
                continue

        if args.backbone == "tlite":
            count = tlite.complete_tlite.count

        print("APICalls for search:\t", count)

        meta_file.write("\n")

        searched_score = self.test(self.result_candidate, args)

        meta_file.write("Testing .... \n")
        if args.print_orig:
            print("Task:\t", chosen_task_name)
            print("Original Instruction:\t", self.original_candidate)
            orig_score = self.score(self.original_candidate, "test", args=args)
            print("Original Accuracy:\t", str(orig_score))
            meta_file.write("Original Accuracy:\t" + str(orig_score) + "\n")

        if self.result_candidate == self.original_candidate:
            print("No viable candidate found!")
            meta_file.write("No viable candidate found!\n")
            print("APICalls:\t", count)
            meta_file.write("APICalls:\t" + str(count) + "\n")
            exit()

        print("Accuracy after search:\t", str(searched_score))
        print("Instruction after search:\t", self.result_candidate)
        meta_file.write("Instruction after search:\t" + self.result_candidate + "\n")
        meta_file.write("Accuracy after search:\t" + str(searched_score) + "\n")
        print("APICalls:\t", count)
        meta_file.write("APICalls:\t" + str(count) + "\n")

    def test(self, instruction, args):

        print("\nTesting .... ")

        searched_score = self.score(instruction, "test", write=args.write_preds, args=args)

        return searched_score
