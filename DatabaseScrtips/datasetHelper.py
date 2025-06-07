import csv
csvPath = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/test.CSV'
selectedDataset = None
selectedDataset =  'busuclm' #'tnscui'#'stu' #'breast' #'tn3k'#'tg3k'#'tnscui'
def getTextSample(dataset=None):
    textCSV = {}
    with open(csvPath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if dataset == None or row['dataset'] == dataset:
                textCSV[str(row['image_name'])] =\
                        {
                        'promt_text': row['label_name'],
                        'bbox': [
                            int(row['bbox_x']),
                            int(row['bbox_y']),
                            int(row['bbox_width']),
                            int(row['bbox_height'])
                        ],
                        'image_size': [
                            int(row['image_width']),
                            int(row['image_height'])
                        ],
                        # 'mask_path': row['mask_path']
                    }
    return textCSV
textCSV = getTextSample(selectedDataset)