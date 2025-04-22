import pdb
import random

from utils.model_loader import ModelLoader

model_loader = ModelLoader()

tokenizer = model_loader.tokenizer


def lowercase_list(lst):
    return [el.lower() for el in lst]


def one_token(label):
    return tokenizer.decode(tokenizer.encode(label, return_tensors="pt")[0][0])


def encode_instruction(
    data=model_loader.sample_data,
    instruction_structure=[
        "Definition",
        "Prompt",
        "Things to Avoid",
        "Emphasis & Caution",
        "Negative Examples Full Explanations",
        "Positive Examples Full Explanations",
    ],
    number_of_examples=0,
    number_of_instances=100,
    null_word=None,
    data_seed=0,
    modified={},
    args=None,
):
    random.seed(0)  # Ensure the same test set

    # // labels = list(set([data["Instances"][i]["output"][0] for i in range(len(data["Instances"]))]))
    label_tokens = [label_id.item() for _, _, label_id in data]
    # labels_unpadded = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
    # label_tokens_unpadded = tokenizer(labels_unpadded)["input_ids"]
    labels = list(set(label_tokens))  # * будем считать лейблами label_id
    labels.sort()

    # *assert len(labels) < 25, "Check {} is a classification task.".format(data.name)
    instances_per_label = number_of_instances // len(labels)
    remainder = number_of_instances % len(labels)
    instance_pools = {label: {"indices": []} for label in labels}
    # todo можно соптимизировать: мы уже считали лейблы
    for i in range(len(data)):
        label = data[i][2].item()
        instance_pools[label]["indices"].append(i)
    remaining = 0
    test_pools = {}

    for el, label in enumerate(labels):
        # leave out some examples for Definition + Examples (hard-coded)
        if len(instance_pools[label]["indices"]) >= 4 + instances_per_label:
            num = instances_per_label
            if el < remainder:
                num += 1

            test_pools[label] = random.sample(instance_pools[label]["indices"], num)
            instance_pools[label]["indices"] = [
                i for i in instance_pools[label]["indices"] if i not in test_pools[label]
            ]
        else:
            num = len(instance_pools[label]["indices"]) - 4
            remaining += instances_per_label - num

            test_pools[label] = random.sample(instance_pools[label]["indices"], num)
            instance_pools[label]["indices"] = [
                i for i in instance_pools[label]["indices"] if i not in test_pools[label]
            ]

    all_remaining_indices = []
    remaining = number_of_instances - sum([len(t) for t in test_pools.values()])
    for label in labels:
        all_remaining_indices.extend(instance_pools[label]["indices"])
    remaining_test = random.sample(all_remaining_indices, remaining)

    for t in remaining_test:
        label = data[t][2].item()
        test_pools[label].append(t)
        instance_pools[label]["indices"].remove(t)

    indexlist = []
    for label in labels:
        indexlist.extend(test_pools[label])
    assert len(indexlist) == number_of_instances, pdb.set_trace()

    random.seed(data_seed)
    # * пока что думаем что zero shot
    chosen_examples = []
    if number_of_examples > 0:
        if number_of_examples == -1:
            total_num_examples = 1
        else:
            total_num_examples = number_of_examples * len(labels)
        pos_examples = {label: [] for label in labels}
        for eg in data["Positive Examples"]:
            label = eg["output"]
            try:
                pos_examples[label].append(eg)
            except:  # noqa: E722
                pdb.set_trace()
        for label in labels:
            for id in instance_pools[label]["indices"]:
                inst = data["Instances"][id]
                inst["output"] = inst["output"][0]
                pos_examples[label].append(inst)

    # total_num_examples = number_of_examples * len(labels)  # todo
    # if number_of_examples > 0:
    #     for label in labels:
    #         chosen_examples.extend(random.sample(pos_examples[label], number_of_examples))
    # elif number_of_examples == -1:
    #     label = random.sample(labels, 1)
    #     chosen_examples.extend(random.sample(pos_examples[label], number_of_examples))
    # assert len(chosen_examples) == total_num_examples
    # random.shuffle(chosen_examples)

    generic_instruction = ""
    for i in instruction_structure:  # ! тут непонятно что происходит.
        if i not in [
            "Positive Examples Full Only",
            "Positive Examples Full Explanations",
            "Negative Examples Full Explanations",
        ]:
            if data[i] != "-":
                if i in modified.keys():
                    data[i] = modified[i]
                data[i] = data[i].replace("\n" + "Things to avoid: -", "")
                data[i] = data[i].replace("\n" + "Emphasis & Caution: -", "")
                if generic_instruction == "":
                    generic_instruction = generic_instruction + i + ": " + data[i].strip()
                else:
                    generic_instruction = generic_instruction + "\n" + i + ": " + data[i].strip()
        elif i == "Positive Examples Full Only":
            for j in range(total_num_examples):
                if "examples" in modified.keys():
                    if generic_instruction != "":
                        generic_instruction = (
                            generic_instruction
                            + "\n"
                            + "input: "
                            + modified["examples"][j]["input"]
                            + "\n"
                            + "output: "
                            + one_token(modified["examples"][j]["output"])
                        )
                    else:
                        generic_instruction = (
                            generic_instruction
                            + "input: "
                            + modified["examples"]["input"]
                            + "\n"
                            + "output: "
                            + one_token(modified["examples"][j]["output"])
                        )
                else:
                    if generic_instruction != "":
                        generic_instruction = (
                            generic_instruction
                            + "\n"
                            + "input: "
                            + chosen_examples[j]["input"]
                            + "\n"
                            + "output: "
                            + one_token(chosen_examples[j]["output"])
                        )
                    else:
                        generic_instruction = (
                            generic_instruction
                            + "input: "
                            + chosen_examples[j]["input"]
                            + "\n"
                            + "output: "
                            + one_token(chosen_examples[j]["output"])
                        )

        elif i == "Positive Examples Full Explanations":  # This mode of Natural Instructions not supported
            assert False

        elif i == "Negative Examples Full Explanations":  # This mode of Natural Instructions not supported
            assert False

    promptlist = []
    answerlist = []

    for i in range(number_of_instances):
        input = tokenizer.decode(data[indexlist[i]][0], skip_special_tokens=True)  # ? все ли тут правильно
        if null_word is None:
            if "input" in modified.keys():
                if generic_instruction != "":
                    prompt = generic_instruction + "\n" + "input: " + input + " " + modified["input"] + "\n" + "output:"
                else:
                    prompt = "input: " + input + "\n" + "output:"
            else:
                if generic_instruction != "":
                    prompt = generic_instruction + "\n" + "input: " + input + "\n" + "output:"
                else:
                    prompt = "input: " + input + "\n" + "output:"
        else:
            if generic_instruction != "":
                prompt = generic_instruction + "\n" + "input: " + null_word + "\n" + "output:"
            else:
                prompt = "input: " + null_word + "\n" + "output:"
        # if "Completion" in labels[0]: #? ??? откуда там бывает completion
        #    prompt = prompt + " Completion"
        promptlist.append(prompt)
        output = tokenizer.decode(data[indexlist[i]][2], skip_special_tokens=True)
        answer = output.strip(".").replace("Completion ", "")
        answer = one_token(answer)
        answerlist.append(answer)

    return promptlist, answerlist, indexlist


def training_encode_instruction(
    # todo тут то же самое
    data=model_loader.sample_data,
    instruction_structure=[
        "Definition",
        "Prompt",
        "Things to Avoid",
        "Emphasis & Caution",
        "Negative Examples Full Explanations",
        "Positive Examples Full Explanations",
    ],
    number_of_examples=0,
    number_of_instances=100,
    null_word=None,
    data_seed=0,
    modified={},
    args=None,
):
    random.seed(0)  # Ensure the same test set
    labels = [label_id.item() for _, _, label_id in data]
    labels.sort()
    assert len(labels) < 25, "Check {} is a classification task.".format(data.name)
    instances_per_label = number_of_instances // len(labels)
    remainder = number_of_instances % len(labels)
    instance_pools = {label: {"indices": []} for label in labels}
    for i in range(len(data)):
        label = data[i][2].item()
        instance_pools[label]["indices"].append(i)
    remaining = 0
    test_pools = {}

    for el, label in enumerate(labels):
        if len(instance_pools[label]["indices"]) >= 4 + instances_per_label:  # see comment in function above
            num = instances_per_label
            if el < remainder:
                num += 1

            test_pools[label] = random.sample(instance_pools[label]["indices"], num)
            instance_pools[label]["indices"] = [
                i for i in instance_pools[label]["indices"] if i not in test_pools[label]
            ]
        else:
            num = len(instance_pools[label]["indices"]) - 4
            remaining += instances_per_label - num

            test_pools[label] = random.sample(instance_pools[label]["indices"], num)
            instance_pools[label]["indices"] = [
                i for i in instance_pools[label]["indices"] if i not in test_pools[label]
            ]

    all_remaining_indices = []
    remaining = number_of_instances - sum([len(t) for t in test_pools.values()])
    for label in labels:
        all_remaining_indices.extend(instance_pools[label]["indices"])
    remaining_test = random.sample(all_remaining_indices, remaining)

    for t in remaining_test:
        label = data[t][2].item()
        test_pools[label].append(t)
        instance_pools[label]["indices"].remove(t)

    indexlist = []
    for label in labels:
        indexlist.extend(test_pools[label])
    assert len(indexlist) == number_of_instances, pdb.set_trace()

    random.seed(data_seed)
    chosen_examples = []
    # * тут идет few shot, пока не трогаю
    if number_of_examples > 0:
        if number_of_examples == -1:
            total_num_examples = 1
        else:
            total_num_examples = number_of_examples * len(labels)
        pos_examples = {label: [] for label in labels}
        for eg in data["Positive Examples"]:
            label = eg["output"]
            pos_examples[label].append(eg)
        for label in labels:
            for id in instance_pools[label]["indices"]:
                inst = data["Instances"][id]
                inst["output"] = inst["output"][0]
                pos_examples[label].append(inst)

        chosen_examples = []
        if number_of_examples > 0:
            for label in labels:
                chosen_examples.extend(random.sample(pos_examples[label], number_of_examples))
        elif number_of_examples == -1:
            label = random.sample(labels, 1)
            chosen_examples.extend(random.sample(pos_examples[label], number_of_examples))
        assert len(chosen_examples) == total_num_examples
        random.shuffle(chosen_examples)

    train_indexlist = list(range(len(data)))
    train_indexlist = [i for i in train_indexlist if i not in indexlist and data[i] not in chosen_examples]

    dev_len = round(0.1 * len(train_indexlist))
    dev_indexlist = random.sample(train_indexlist, dev_len)
    train_indexlist = [i for i in train_indexlist if i not in dev_indexlist]

    # ! снова непонятно что
    generic_instruction = ""
    for i in instruction_structure:
        if i not in [
            "Positive Examples Full Only",
            "Positive Examples Full Explanations",
            "Negative Examples Full Explanations",
        ]:
            if data[i] != "-":
                if i in modified.keys():
                    data[i] = modified[i]
                data[i] = data[i].replace("\n" + "Things to avoid: -", "")
                data[i] = data[i].replace("\n" + "Emphasis & Caution: -", "")
                # pdb.set_trace()
                if generic_instruction == "":
                    generic_instruction = generic_instruction + i + ": " + data[i].strip()
                else:
                    generic_instruction = generic_instruction + "\n" + i + ": " + data[i].strip()
        elif i == "Positive Examples Full Only":
            for j in range(total_num_examples):
                if generic_instruction != "":
                    generic_instruction = (
                        generic_instruction
                        + "\n"
                        + "input: "
                        + chosen_examples[j]["input"]
                        + "\n"
                        + "output: "
                        + one_token(chosen_examples[j]["output"])
                    )
                else:
                    generic_instruction = (
                        generic_instruction
                        + "input: "
                        + chosen_examples[j]["input"]
                        + "\n"
                        + "output: "
                        + one_token(chosen_examples[j]["output"])
                    )

        elif i == "Positive Examples Full Explanations":  # This mode of Natural Instructions not supported
            assert False

        elif i == "Negative Examples Full Explanations":  # This mode of Natural Instructions not supported
            assert False

    promptlist = []
    answerlist = []

    for i in range(number_of_instances):
        input = tokenizer.decode(data[indexlist[i]][0], skip_special_tokens=True)
        if null_word is None:
            if generic_instruction != "":
                prompt = generic_instruction + "\n" + "input: " + input + "\n" + "output:"
            else:
                prompt = "input: " + input + "\n" + "output:"
        else:
            if generic_instruction != "":
                prompt = generic_instruction + "\n" + "input: " + null_word + "\n" + "output:"
            else:
                prompt = "input: " + null_word + "\n" + "output:"
        # if "Completion" in labels[0]: # * тоже непонятно
        #    prompt = prompt + " Completion"
        promptlist.append(prompt)
        output = tokenizer.decode(data[indexlist[i]][2], skip_special_tokens=True)
        answer = output.strip(".").replace("Completion ", "")
        answer = one_token(answer)
        answerlist.append(answer)

    train_promptlist = []
    train_answerlist = []

    for i in range(len(train_indexlist)):
        input = tokenizer.decode(data[train_indexlist[i]][0], skip_special_tokens=True)
        if null_word is None:
            if generic_instruction != "":
                prompt = generic_instruction + "\n" + "input: " + input + "\n" + "output:"
            else:
                prompt = "input: " + input + "\n" + "output:"
        else:
            if generic_instruction != "":
                prompt = generic_instruction + "\n" + "input: " + null_word + "\n" + "output:"
            else:
                prompt = "input: " + null_word + "\n" + "output:"
        # if "Completion" in labels[0]:
        #    prompt = prompt + " Completion"
        train_promptlist.append(prompt)
        output = tokenizer.decode(data[train_indexlist[i]][2], skip_special_tokens=True)
        train_answer = output.strip(".").replace("Completion ", "")
        train_answer = one_token(train_answer)
        train_answerlist.append(train_answer)

    dev_promptlist = []
    dev_answerlist = []

    for i in range(len(dev_indexlist)):
        input = tokenizer.decode(data[dev_indexlist[i]][0], skip_special_tokens=True)
        if null_word is None:
            if generic_instruction != "":
                prompt = generic_instruction + "\n" + "input: " + input + "\n" + "output:"
            else:
                prompt = "input: " + input + "\n" + "output:"
        else:
            if generic_instruction != "":
                prompt = generic_instruction + "\n" + "input: " + null_word + "\n" + "output:"
            else:
                prompt = "input: " + null_word + "\n" + "output:"
        # if "Completion" in labels[0]:
        #    prompt = prompt + " Completion"
        dev_promptlist.append(prompt)
        output = tokenizer.decode(data[dev_indexlist[i]][2], skip_special_tokens=True)
        dev_answer = output.strip(".").replace("Completion ", "")
        dev_answer = one_token(dev_answer)
        dev_answerlist.append(dev_answer)
    return (
        promptlist,
        answerlist,
        indexlist,
        train_promptlist,
        train_answerlist,
        train_indexlist,
        dev_promptlist,
        dev_answerlist,
        dev_indexlist,
    )
