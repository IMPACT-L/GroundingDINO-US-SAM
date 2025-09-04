import csv
import shutil
import os

desDir = '../Grounding-Sam-Ultrasound/multimodal-data'
csvFileName = ''
def create_dataset(data_list, output_type):
    init()
    annotation_file = f'{desDir}/{output_type}{csvFileName}.CSV'

    with open(annotation_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data_list:
            try:
                
                image_name = os.path.basename(row[5])
                extra = ''
                if len(row) == 11 and row[10] > 0:
                    extra=row[10]
                if 'test' in output_type:
                   

                    shutil.copy2(row[5], f'{desDir}/{output_type}_image/{row[9]}{extra}_{image_name.lower()}')
                    mask_path = f'{desDir}/{output_type}_mask/{row[9]}{extra}_{image_name.lower()}'
                    shutil.copy2(row[8], mask_path)
                else:
                    shutil.copy2(row[5], f'{desDir}/{output_type}/{row[9]}{extra}_{image_name.lower()}')
                    mask_path = f'{desDir}/{output_type}/{row[9]}{extra}_{image_name.lower()}'

                row[5] = f"{row[9]}{extra}_{row[5].split('/')[-1]}".lower()
                row[8] = row[9]
                # = f"{mask_path.split('/')[-1]}".lower()
                row[0] = row[0].lower()

                writer.writerow(row[:9])
                print(f'Saved: {image_name.lower()}')
            except Exception as e:
                print(f"Error processing {row}: {str(e)}")

def init():
    os.makedirs(f'{desDir}/train', exist_ok=True)
    os.makedirs(f'{desDir}/val', exist_ok=True)
    os.makedirs(f'{desDir}/test_image', exist_ok=True)
    os.makedirs(f'{desDir}/test_mask', exist_ok=True)

    header = [
        'label_name', 'bbox_x', 'bbox_y', 
        'bbox_width', 'bbox_height', 
        'image_name', 'image_width', 'image_height',
        'dataset'
    ]
    # 'mask_path',
    for output_type in ['test','train', 'val']:
        file_path = f'{desDir}/{output_type}{csvFileName}.CSV'

        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
        else:
            with open(file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                existing_header = next(reader, [])
                if existing_header != header:
                    with open(file_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(header)
