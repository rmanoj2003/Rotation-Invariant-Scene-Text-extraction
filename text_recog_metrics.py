import os
import torch
import string
import validators
import time
from utils.infer_utils import TokenLabelConverter, NormalizePAD, ViTSTRFeatureExtractor, get_args
from Levenshtein import distance as levenshtein_distance
from Levenshtein import ratio as levenshtein_ratio

def img2text(model, images, converter):
    pred_strs = []
    with torch.no_grad():
        for img in images:
            pred = model(img, seqlen=converter.batch_max_length)
            _, pred_index = pred.topk(1, dim=-1, largest=True, sorted=True)
            pred_index = pred_index.view(-1, converter.batch_max_length)
            length_for_pred = torch.IntTensor([converter.batch_max_length - 1])
            pred_str = converter.decode(pred_index[:, 1:], length_for_pred)
            pred_EOS = pred_str[0].find('[s]')
            pred_str = pred_str[0][:pred_EOS]
            pred_strs.append(pred_str)
    return pred_strs

def infer(args, files, base_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    converter = TokenLabelConverter(args)
    args.num_class = len(converter.character)
    extractor = ViTSTRFeatureExtractor()

    images = []
    for f in files:
        img_path = os.path.join(base_dir, f)
        img = extractor(img_path)
        if args.gpu:
            img = img.to(device)
        images.append(img)

    if args.quantized:
        if args.rpi:
            backend = "qnnpack"
        else:
            backend = "fbgemm"
        torch.backends.quantized.engine = backend

    if validators.url(args.model):
        checkpoint = args.model.rsplit('/', 1)[-1]
        torch.hub.download_url_to_file(args.model, checkpoint)
    else:
        checkpoint = args.model

    if args.quantized:
        model = torch.jit.load(checkpoint)
    else:
        model = torch.load(checkpoint)

    if args.gpu:
        model.to(device)

    model.eval()
    pred_strs = img2text(model, images, converter)
    return pred_strs

def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            image_file, gt_text = line.strip().split(', ')
            dataset.append((image_file, gt_text.strip('"')))
    return dataset

def calculate_edit_distance(pred_texts, gt_texts):
    total_distance = 0
    for pred, gt in zip(pred_texts, gt_texts):
        for pred_token, gt_token in zip(pred.split(), gt.split()):
            total_distance += levenshtein_ratio(pred_token.lower(), gt_token.lower())
    average_distance = total_distance / len(gt_texts)
    return average_distance

def calculate_accuracy(pred_texts, gt_texts, threshold=0.65):
    success_count = 0
    for pred, gt in zip(pred_texts, gt_texts):
        for pred_token, gt_token in zip(pred.split(), gt.split()):
            ratio = levenshtein_ratio(pred_token.lower(), gt_token.lower())
            if ratio >= threshold:
                success_count += 1
    accuracy = success_count / len(gt_texts)
    return accuracy

if __name__ == '__main__':
    args = get_args()
    args.character = string.printable[:-6]

    dataset_path = r'D:\Projects\RPP\Sem 7\RISTE\data\raw\icdar\ch4_test_word_images_gt\Challenge4_Test_Task3_GT.txt'
    base_image_dir = r'D:\Projects\RPP\Sem 7\RISTE\data\raw\icdar\ch4_test_word_images_gt'

    dataset = load_dataset(dataset_path)

    image_files, gt_texts = zip(*dataset)
    pred_texts = infer(args, image_files, base_image_dir)

    average_edit_distance = calculate_edit_distance(pred_texts, gt_texts)
    print(f"Average Levenshtein Ratio: {average_edit_distance:.4f}")

    accuracy = calculate_accuracy(pred_texts, gt_texts)
    print(f"Accuracy: {accuracy:.4f}")