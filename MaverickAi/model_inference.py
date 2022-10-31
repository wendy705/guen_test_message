from transformers import pipeline, BertTokenizer
import numpy as np

labels = {0: 'J80', 1: 'E039', 2: 'I4891', 3: 'F319', 4: 'E119', 5: 'F0280', 6: 'I609', 7: 'R65.21', 8: 'K7030', 9: 'K766', 10: 'M810', 11: 'B182', 12: 'D696', 13: 'I469', 14: 'N186', 15: 'J449', 16: 'I2699', 17: 'Z79.4', 18: 'F10239', 19: 'I25.2', 20: 'E6601', 21: 'A419', 22: 'I714', 23: 'R570', 24: 'J15211', 25: 'I472', 26: 'I10', 27: 'E780', 28: 'F329', 29: 'E46', 30: 'K219', 31: 'I6529', 32: 'I619', 33: 'I6350', 34: 'J9620', 35: 'G936', 36: 'M069', 37: 'N189', 38: 'I739', 39: 'N179', 40: 'I200', 41: 'I214', 42: 'I509', 43: 'C7931', 44: 'I2510', 45: 'I129', 46: 'J690', 47: 'R569', 48: 'C787', 49: 'F341'}


def inference(sentence):

    tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', do_lower_case=+True)
    model_id = "Guen/BertFinetunedMIMICIII"
    classifier = pipeline("text-classification", model=model_id, tokenizer=tokenizer)

    preds = classifier(sentence, return_all_scores=True)
    maxProb = max(preds[0], key=lambda x: x['score'])
    label = maxProb['label']
    label = label.replace('LABEL_', '')
    label = int(label)
    inv_label = labels[label]
    return(inv_label)




