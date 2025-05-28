# loader = ModelLoader("sst-2")
# loader.seed_everything()
# print(
#     loader.evaluator.evaluate_vllm(
#         model=loader.model,
#         tokenizer=loader.tokenizer,
#         eval_ds=loader.load_data(prompt="Please perform Sentiment Classification task.", split="test"),
#         batch_size=16,
#         model_generate_args=loader.model_generate_args,
#     )
# )
# loader.destroy()
import torch

# project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
# sys.path.append(project_root)
# from utils.model_loader import ModelLoader  # noqa 402

torch.cuda.empty_cache()
