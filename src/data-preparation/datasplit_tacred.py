import sys
import os
import json
import random
import configparser
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))


PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"
print(PREFIX_PATH)

def read_json(path):
    """ Read a json file from the given path."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    """ Write a json file to the given path."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_prompt(sentence, head, tail, relations, prompt_type):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """
    relations = ", ".join([relation for relation in relations])

    # template_zero_shot = """Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n""" +\
    #                     """ Question: What is the relation type between tail and head entities according to given relationships below in the following sentence?\n""" +\
    #                     """ Query Sentence: """ + str(sentence)+ """\n""" +\
    #                     """ head: """ + head + """. \n""" +\
    #                     """ tail: """ + tail + """. \n""" +\
    #                     """ Relation types: """ + relations + """. \n""" +\
    #                     """ output format: relation_type"""
    if not prompt_type:
        template2_zero_shot = """Sentence: """ + str(sentence)+ """\n""" +\
                            """ What is the relation type between """+head+""" and """+tail+"""  according to given relation types below in the sentence?\n""" +\
                            """ Relation types: """ + relations + """. \n""" +\
                            """ Answer:"""
    else:
        template2_zero_shot = """Sentence: """ + str(sentence)+ """\n""" +\
                              """What is the relation type between """+head+""" and """+tail+"""  according to given relation types below in the sentence?\n""" +\
                              """Relation types: """ + relations + """. \n"""
                             
    return template2_zero_shot

def prepare_instructions(task_train_data, task_dev_data, task_test_data, relations, task_relations, task_id, run_id, out_folder, prompt_type=False):
    """ Prepare instructions for each task in each run
    Args:
        task_train_data: train data for the task
        task_dev_data: dev data for the task
        task_test_data: test data for the task
        relations: all relations
        task_relations: relations for the task
        task_id: task id
        run_id: run id
        out_folder: path to save instructions
    Return: task_dev_data, task_test_data
    """
    data = {"train":task_train_data, "dev":task_dev_data}
    selected_data = []

    for key, value in data.items():
    
        out_file_path = out_folder+"train/run_{0}/task{1}/{2}_1.json".format(run_id, task_id, key)
        prompts = []
        selected_data = []
        for relation in task_relations:
            relation_data = [line for line in value if relation == line['relation']]
            # print(relation_data[0])
            if key == "train":
                if len(relation_data) > 320:
                    ids = [line['id'] for line in relation_data]
                    selected_ids = random.sample(ids, 320)
                    relation_data = [ line for line in relation_data if line["id"] in selected_ids]
                    selected_data.extend(relation_data)
                else:
                    selected_data.extend(relation_data)
                selected_data.extend(relation_data)
            if key == "dev":
                if len(relation_data)>40:
                    ids = [line['id'] for line in relation_data]
                    selected_ids = random.sample(ids, 40)
                    relation_data = [ line for line in relation_data if line["id"] in selected_ids]
                    selected_data.extend(relation_data)
                else:
                    selected_data.extend(relation_data)
                selected_data.extend(relation_data)
        if key == "dev":
            task_dev_data = selected_data
        else:
            task_train_data = selected_data

        for line in selected_data:
            # input = {"id":line["id"],"sentence":line['sentence'],"subject": line['subject'], "object": line['object'], "subject_type":line["subject_type"], "object_type":line["object_type"],  "relation": line['relation']}
            input = {"prompt":get_prompt(line['sentence'],line['subject'], line['object'], relations, prompt_type), "relation":line['relation'], "sentence":line['sentence'], "subject": line['subject'], "object": line['object'], "subject_type":line["subject_type"],"object_type":line["object_type"]}
            prompts.append(input)
            
        
        # print(len(relations))
        write_json(out_file_path, prompts)
        #memory recoding
        if key == "dev":
            for i in range(task_id+1, 11):
                out_file_path = out_folder+"train/run_{0}/task_memory_{1}/{2}_{3}.json".format(run_id, i, key, task_id)
                write_json(out_file_path, prompts)

    prompts = []
    out_test_file_path = out_folder+"test/run_{0}/task{1}/test_1.json".format(run_id, task_id)
    selected_test_data = []

    for relation in task_relations:
        test_relation_data = [line for line in task_test_data if line['relation']==relation]

        if len(test_relation_data)>40:
            ids = [line['id'] for line in test_relation_data]
            selected_ids = random.sample(ids, 40)
            test_relation_data = [line for line in task_test_data if line["id"] in selected_ids]

        selected_test_data.extend(test_relation_data)    
    task_test_data = selected_test_data

        
    for line in selected_test_data:
        # input = {"id":line["id"],"sentence":line['sentence'],"subject": line['subject'], "object": line['object'], "subject_type":line["subject_type"],"object_type":line["object_type"], "relation": line['relation']}
        input = {"prompt":get_prompt(line['sentence'],line['subject'], line['object'], relations, prompt_type), "relation":line['relation'], "sentence":line['sentence'], "subject": line['subject'], "object": line['object'], "subject_type":line["subject_type"],"object_type":line["object_type"]}
        prompts.append(input)

    write_json(out_test_file_path, prompts)
    return task_dev_data, task_test_data


def main(all_train_data_path, all_dev_data_path, all_test_data_path, all_tasks_path, out_folder, prompt_type=False):
    """ Main function to prepare instructions for each task in each run
    Args:
        all_train_data_path: path to all train data
        all_dev_data_path: path to all dev data
        all_test_data_path: path to all test data
        all_tasks_path: path to all tasks
        out_folder: path to save instructions

    """
    all_train_data  = read_json(all_train_data_path)
    all_test_data = read_json(all_test_data_path)
    all_dev_data = read_json(all_dev_data_path)
    all_tasks = read_json(all_tasks_path)

    
    for run_id in range(1,6):

        run_name = "run_{0}".format(run_id)
        run_tasks = all_tasks[run_name]

        for task_id in range(1, 11):
            task_name = "task{0}".format(task_id)
            task_relations = run_tasks[task_name]
            file = out_folder+"relations/run_{0}/task{1}.json".format(run_id, task_id)
            write_json(file, task_relations)
            task_train_data = [item for item in all_train_data if item['relation'] in task_relations]
            task_test_data = [item for item in all_test_data if item['relation'] in task_relations]
            task_dev_data = [item for item in all_dev_data if item['relation'] in task_relations]
            test_relations = [item['relation'] for item in all_test_data if item['relation'] in task_relations]
           
            prepare_instructions(task_train_data, task_dev_data, task_test_data, task_relations, task_relations,task_id, run_id, out_folder, prompt_type)

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read(PREFIX_PATH+'config.ini')
    all_train_data  = "/Users/sefika/phd_projects/neurogenesis-cre/data/tacred/data/train.json"
    all_test_data = "/Users/sefika/phd_projects/neurogenesis-cre/data/tacred/data/test.json"
    all_dev_data = "/Users/sefika/phd_projects/neurogenesis-cre/data/tacred/data/dev.json"
    all_tasks = "/Users/sefika/phd_projects/neurogenesis-cre/data/tacred/final/tacred_tasks.json"
    out_folder =  "tacred/final/"
    main(all_train_data, all_dev_data, all_test_data, all_tasks, out_folder)