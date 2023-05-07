from datasets import load_dataset_builder, load_dataset
import logging 

def inspect():
    langs = ['amharic','english','hausa','swahili','yoruba','igbo']

    for lang in langs:
        ds_builder = load_dataset_builder("csebuetnlp/xlsum",lang)
        
        desc = ds_builder.info.description
        
        feat = ds_builder.info.features
        
        return desc,feat
    
def load():
    try:
        langs = ['amharic','hausa','swahili','yoruba','igbo']
        
        for lang in langs:

            dataset = load_dataset("csebuetnlp/xlsum", lang ,split="train")
            #for split, data in dataset.items():
            dataset.to_csv(f"{lang}.csv", index = None)
            #dataset.save_to_disk(lang) 
            #return dataset
    except Exception as ex:
        logging.debug(ex)

if __name__ == '__main__':
    #print(inspect())
    load()