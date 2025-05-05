import argparse
import os
import sys

from config import get_cfg_default

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(project_root)

from utils import set_random_seed, setup_logger  # noqa 402
from utils.model_loader import ModelLoader  # noqa 402


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.meta_dir:
        cfg.META_DIR = args.meta_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.data_seed >= 0:
        cfg.DATA_SEED = args.data_seed

    if args.train_seed >= 0:
        cfg.TRAIN_SEED = args.train_seed


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.GA = CN()
    cfg.TRAINER.GA.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    print("Started main")
    cfg = setup_cfg(args)
    print("Setting fixed data_seed: {}, train_seed: {}".format(cfg.DATA_SEED, cfg.TRAIN_SEED))
    set_random_seed(cfg.DATA_SEED, cfg.TRAIN_SEED)
    setup_logger(cfg.META_DIR)
    # //print_args(args, cfg)
    print("Collecting env info ...")
    # // print("** System info **\n{}\n".format(collect_env_info()))
    # if args.backbone == "tlite":
    #     import utils.tlite as tlite

    #     construct_instruction_prompt = tlite.construct_instruction_prompt

    data_seed = args.data_seed
    train_seed = args.train_seed
    data_base_path = args.data_dir
    print(data_base_path)
    num_compose = args.num_compose
    num_candidates = args.num_candidates
    num_steps = args.num_iter
    num_tournaments = args.tournament_selection
    patience = args.patience
    # edit_operations = args.edits
    backbone = args.backbone

    # _, task_labels, _ = construct_instruction_prompt(
    #     mode="No Instructions",
    #     num_shots=num_shots,
    #     num_test_instances=num_samples,
    #     data_seed=data_seed,
    #     args=args,
    # ) # * давай заменим на джсон
    if args.bench_name != "":
        loader = ModelLoader(args.task_name, args.bench_name)
    else:
        loader = ModelLoader(args.task_name)
    loader.seed_everything()
    task_labels = loader.labels
    print("Running Experiment for: ", args.task_name)
    print("Task Labels: ", task_labels)
    args.task_labels = task_labels

    instruction = loader.base_prompt
    print("Original Instruction: ", instruction)
    # // instruction[0].replace("\n" + "Things to avoid: -", "")
    # // print("Instruction Edit1: ", instruction)
    # // instruction = instruction[0].replace("\n" + "Emphasis & Caution: -", "")
    # // print("Instruction Edit2: ", instruction)  # ???
    if args.agnostic:
        # * агностик = общий промпт для всех задач
        instruction = (
            "You will be given a task. Read and understand the task carefully, and appropriately "
            "answer '{}' or '{}'.".format(task_labels[0], task_labels[1])
        )
    loader.print_gpu_memory()
    from trainers import GA_trainer, HC_trainer, HS_trainer, TB_trainer  # noqa 402

    if args.algorithm == "ga":
        trainer = GA_trainer.GA_trainer(
            num_steps, patience, train_seed, data_seed, num_compose, num_candidates, num_tournaments, backbone
        )
    elif args.algorithm == "hc":
        trainer = HC_trainer.HC_trainer(
            num_steps, patience, train_seed, data_seed, num_compose, num_candidates, backbone
        )
    elif args.algorithm == "hs":
        trainer = HS_trainer.HS_trainer(
            num_steps, patience, train_seed, data_seed, num_compose, num_candidates, backbone
        )
    elif args.algorithm == "tabu":
        trainer = TB_trainer.TB_trainer(
            num_steps, patience, train_seed, data_seed, num_compose, num_candidates, backbone
        )
    else:
        raise ValueError("Invalid algorithm type.")

    if args.eval_only:
        print("Testing the performance of the loaded model...")
        trainer.load(args.model_dir)
        instruction = trainer.state["result_candidate"]
        accuracy = trainer.test(instruction, args)
        print("eval_acc: {}".format(accuracy))
    else:
        if not args.no_train:
            try:
                trainer.train(instruction, args)
            except:
                meta_test_path = os.path.join(args.meta_test_dir, args.meta_test_name)
                meta_test_file = open(meta_test_path, "a")
                meta_test_file.write(f"{args.task_name} recieved an error\n")
    loader.destroy()


if __name__ == "__main__":
    print("Started")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="sst-2", help="Name of the task")
    parser.add_argument("--bench-name", default="", help="Name of the benchmark bbh/nat instr")
    parser.add_argument("--mode", default="Instruction Only", help="Mode of instructions/prompts")
    parser.add_argument("--model-name", default="text-babbage-001", help="Name of used model")
    parser.add_argument("--num-shots", default=2, type=int, help="Number of examples in the prompt if applicable")
    parser.add_argument("--batch-size", default=4, type=int, help="Batch size")
    parser.add_argument("--task-idx", default=2, type=int, help="The index of the task based on the array in the code")
    parser.add_argument(
        "--data-seed", default=42, type=int, help="Seed that changes score dataset by sampling examples"
    )
    parser.add_argument(
        "--train-seed", type=int, help="Seed that changes the sampling of edit operations (search seed)"
    )
    parser.add_argument("--num-compose", default=1, type=int, help="Number of edits composed to get one candidate")
    parser.add_argument("--num-train", default=100, type=int, help="Number of examples in score set")
    parser.add_argument("--level", default="phrase", help="Level at which edit operations occur")
    parser.add_argument(
        "--simulated-anneal",
        action="store_true",
        default=False,
        help="Runs simulated anneal if candidate scores <= base score",
    )
    parser.add_argument(
        "--agnostic", action="store_true", default=False, help="Uses template task-agnostic instruction"
    )
    parser.add_argument(
        "--print-orig",
        action="store_true",
        default=False,
        help="Print original instruction and evaluate its performance",
    )
    parser.add_argument("--write-preds", action="store_true", default=False, help="Store predictions in a .json file")
    parser.add_argument("--data-dir", default="./natural-instructions-2.6/tasks/", help="Path to the dataset")
    parser.add_argument("--meta-dir", default="logs/", help="Path to store metadata of search")
    parser.add_argument(
        "--meta-test-dir", default="logs_test/", help="Path to store metadata of search for future test"
    )
    parser.add_argument("--meta-name", default="search.txt", help="Path to the file that stores metadata of search")
    parser.add_argument(
        "--meta-test-name",
        default="search_test.txt",
        help="Path to the file that stores metadata of search for future test",
    )
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--patience", default=2, type=int, help="The max patience P (counter)")
    parser.add_argument("--num-candidates", default=5, type=int, help="Number of candidates in each iteration (m)")
    parser.add_argument("--num-iter", default=10, type=int, help="Max number of search iterations")
    parser.add_argument("--key-id", default=0, type=int, help="Use if you have access to multiple Open AI keys")
    parser.add_argument(
        "--edits", nargs="+", default=["del", "swap", "sub", "add"], help="Space of edit ops to be considered"
    )
    parser.add_argument("--tournament-selection", default=3, type=int, help="Number of tournament selections")
    parser.add_argument("--project-name", default="evolutional-prompt", help="Name of the wandb project")
    parser.add_argument("--num-samples", default=100, type=int, help="size of score set, default is 100")
    parser.add_argument(
        "--classification-task-ids",
        default=["019", "001", "022", "050", "069", "137", "139", "195"],
        type=list,
        help="classification tasks",
    )
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--checkpoint-freq", type=int, default=5, help="Checkpoint every N steps.")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="", help="path to config file for dataset setup")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("--backbone", default="gpt3", help="backbone model")
    parser.add_argument("--algorithm", default="ga", help="Searching Algorithms")
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line"
    )
    parser.add_argument("--budget", default=1000, type=int, help="number of the budget of api calls for searching")
    parser.add_argument("--api-idx", type=int, default=0)
    args = parser.parse_args()

    main(args)
