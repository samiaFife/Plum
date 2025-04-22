import heapq
import json
import logging
import math
import os
import random
import re
import sys

import numpy as np
from supar import Parser
from trainers.base_trainer import SimpleTrainer

import utils.tlite as tlite

sys.path.append("..")

logger = logging.getLogger(__name__)


class HS_trainer(SimpleTrainer):

    def __init__(self, maxiter, patience, train_seed, seed, num_compose, num_candidates, backbone):
        super(HS_trainer, self).__init__(maxiter, patience, train_seed, seed, num_compose, num_candidates, backbone)
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
        self.state = {}
        self.W_candidates_m = []
        self.W_scores_m = []

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

    def mutated(self, base_candidate, phrase_lookup, use_add, delete_tracker, edit_operations, args):

        deleted = {}
        added = {}
        logger.debug(f"mutated: base candidate {base_candidate}")
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
                edits = np.random.choice(edit_operations, 1)
            else:
                edits = []
                for n in range(1):
                    edits.append(np.random.choice(edit_operations, self.num_compose))
            print(edits)

            # generate candidates
            candidates = []
            for edit in edits:
                if isinstance(edit, str):
                    candidate, indices = self.perform_edit(edit, base_candidate, phrase_lookup, delete_tracker)
                    print("performed successfully #2")
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

        return candidates, deleted, added

    def generate_candidate(self, ks, HMCR, PAR, edit_operations_small, use_add, delete_tracker, edit_operations, args):
        logger.debug(f" generate_candidate: ks {ks}")
        w_m = []
        w_m_words = []
        deleted = {}
        added = {}

        for j in range(ks):
            idx = np.random.randint(0, len(self.W_candidates))
            w = self.W_candidates[idx]
            logger.debug(f"{j}: w: {w}")
            phrases_pun = self.get_phrase_lookup_pun(w, args)
            logger.debug(f"{j}: phrases_pun: {phrases_pun}")
            w_phrases = list(phrases_pun.values())
            L = len(w_phrases)
            start = math.ceil(j / ks * L)
            end = math.ceil((j + 1) / ks * L) - 1
            logger.debug(f"{j}: start {start} end {end}")
            w_segement = w_phrases[start:end]
            logger.debug(f"{j}: w_segement: {w_segement}")
            w_segement_words = []

            for phrase in w_segement:
                w_segement_words = w_segement_words + self.word_tokenize(phrase)
            w_segement = self.detokenize(w_segement_words)
            logger.debug(f"{j}: w_segment: {w_segement}")

            if HMCR >= np.random.random():
                logger.debug(f"{j}: HMCR")
                if PAR >= np.random.random():
                    logger.debug(f"{j}: PAR")
                    try:
                        phrase_lookup = self.get_phrase_lookup(w_segement, args)
                        logger.debug(f"{j}: phrase lookup #1 {phrase_lookup}")
                        candidate, _ = self.perform_edit(
                            edit_operations_small, w_segement, phrase_lookup, delete_tracker
                        )
                        w_segement = candidate
                        print(w_segement)
                        logger.debug(f"{j}: success")
                    except:
                        print("Error occurs (parser) and skip this mutation")
                        continue
                deleted = {}
                added = {}

            else:
                try:
                    phrase_lookup = self.get_phrase_lookup(w_segement, args)
                    logger.debug(f"{j}: phrase lookup #2: {phrase_lookup}")
                    candidates, deleted, added = self.mutated(
                        w_segement, phrase_lookup, use_add, delete_tracker, edit_operations, args
                    )
                    w_segement = candidates[0]  # multipule edit operations can be implemented if necessary
                    print(w_segement)
                    logger.debug(f"{j}: success")
                except:
                    print("Error occurs (parser) and skip this mutation")
                    continue

            w_m.append(w_segement)
        for segement in w_m:
            w_m_words.extend(self.word_tokenize(segement))
        w_m = self.detokenize(w_m_words)
        return w_m, deleted, added

    def train(self, instruction, args):

        ks = 2
        HMCR = 0.4
        PAR = 0.5
        edit_operations_small = "sub"
        N_H = 10

        meta_path = os.path.join(args.meta_dir, args.meta_name)
        meta_file = open(meta_path, "w+")
        meta_test_path = os.path.join(args.meta_test_dir, args.meta_test_name)
        meta_test_file = open(meta_test_path, "a")
        edit_operations = args.edits
        use_add = "add" in edit_operations

        if "sub" in edit_operations:
            self.if_sub(edit_operations)

        self.init_population(instruction, args)

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

            deleted_candidate = {}
            added_candidate = {}
            self.W_candidates_m = []
            self.W_scores_m = []
            for c in range(args.num_candidates):

                w_m, deleted, added = self.generate_candidate(
                    ks, HMCR, PAR, edit_operations_small, use_add, delete_tracker, edit_operations, args
                )
                print(f"Candidate:\n{w_m}\n")
                w_m_score = self.score(w_m, args=args)
                self.W_candidates_m.append(w_m)
                self.W_scores_m.append(w_m_score)
                print(f"{w_m_score:.6f}")
                deleted_list = []
                added_list = []
                for item in list(deleted.values()):
                    deleted_list.extend(item)
                for item in list(added.values()):
                    added_list.extend(item)
                if not deleted_list == []:
                    deleted_candidate[w_m] = deleted_list
                if not added_list == []:
                    added_candidate[w_m] = added_list

            self.update_result(self.W_candidates_m, self.W_scores_m)

            if self.patience_counter > args.patience:
                print("Ran out of patience")
                meta_file.write("Ran out of patience \n")
                break

            self.W_candidates = self.W_candidates + self.W_candidates_m
            self.W_scores = self.W_scores + self.W_scores_m

            top_N_H_idx_list = heapq.nlargest(N_H, range(len(self.W_scores)), self.W_scores.__getitem__)
            W_candidates_top_N_H = [self.W_candidates[i] for i in top_N_H_idx_list]
            W_scores_top_N_H = [self.W_scores[i] for i in top_N_H_idx_list]
            self.W_candidates = W_candidates_top_N_H
            self.W_scores = W_scores_top_N_H

            for candidate in self.W_candidates:

                if candidate in added_candidate.keys():
                    print("Notice! Prev tracker: ", delete_tracker)
                    for chunk in added_candidate[candidate]:
                        try:
                            delete_tracker.remove(chunk)
                        except:
                            pass
                    print("Notice! New tracker: ", delete_tracker)

                if candidate in deleted_candidate.keys():
                    delete_tracker.extend(deleted_candidate[candidate])

                self.result_candidate = self.detokenize(self.word_tokenize(self.result_candidate))

            # if current_iteration % args.checkpoint_freq == 0:
            # self.get_state(current_iteration, delete_tracker)
            # ckpt_dir = Path(args.output_dir) / "checkpoints"
            # ckpt_dir.mkdir(exist_ok=True)
            # filename = "{}_step{}.pickle".format(self.task_name, current_iteration - 1)
            # ckpt_path = ckpt_dir / filename
            # self.save(ckpt_path)

            if args.backbone == "tlite":
                count = tlite.complete_tlite.count

            if count >= args.budget:
                print("Ran out of budget")
                break

            # if self.patience_counter > args.patience:
            #     print('Ran out of patience')
            #     meta_file.write('Ran out of patience \n')
            #     break
            # elif count >= args.budget:
            #     print('Ran out of budget')
            #     break
            # else:
            #     continue

        if args.backbone == "tlite":
            count = tlite.complete_tlite.count

        print("APICalls for search:\t", count)

        meta_file.write("\n")

        searched_score = self.test(self.result_candidate, args)

        meta_file.write("Testing .... \n")
        if args.print_orig:
            print("Task:\t", self.task_name)
            print("Original Instruction:\t", self.original_candidate)
            orig_score = self.score(self.original_candidate, "test", args=args)
            print(f"Original score:\t {orig_score}")
            meta_file.write(f"Original score: {orig_score}" + "\n")

        if self.result_candidate == self.original_candidate:
            print("No viable candidate found!")
            meta_file.write("No viable candidate found!\n")
            print("APICalls:\t", count)
            meta_file.write("APICalls:\t" + str(count) + "\n")
            exit()

        print(f"After search score:\t {searched_score}")
        print("Instruction after search:\t", self.result_candidate)
        meta_file.write("Instruction after search:\t" + self.result_candidate + "\n")
        meta_file.write(f"After search score:\t {searched_score}" + "\n")
        meta_test_file.write(
            json.dumps({"task": self.task_name, "prompt": self.result_candidate, "metrics": searched_score}) + "\n"
        )
        print("APICalls:\t", count)
        meta_file.write("APICalls:\t" + str(count) + "\n")

    def test(self, instruction, args):

        print("\nTesting .... ")

        searched_score = self.score(instruction, "test", write=args.write_preds, args=args)

        return searched_score
