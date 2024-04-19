from PIL import Image
import os
import xlwt

def count_size(image_path):

    image = Image.open(image_path)
    # Convert to grayscale image
    grayscale_image = image.convert('L')
    # Get the width and height of the image
    width, height = grayscale_image.size

    white_pixel_count = 0
    for y in range(height):
        for x in range(width):
            pixel_value = grayscale_image.getpixel((x, y))
            if pixel_value > 200:
                white_pixel_count += 1
    return white_pixel_count

folder_path = "F:/finalproject/ACDC_dataset/abnormal_enhancement/RV_2"
#save at the xls table
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("Pixel Counts")

sheet.write(0, 0, "Image Name")
sheet.write(0, 1, "White Pixel Count")

row = 1
for filename in os.listdir(folder_path):
    if filename.endswith(('.png')):
        image_path = os.path.join(folder_path, filename)
        white_pixel_count = count_size(image_path)
        sheet.write(row, 0, filename)
        sheet.write(row, 1, white_pixel_count)
        row += 1

# save
xls_file_path = 'F:/finalproject/ACDC_dataset/Features/Enhancement_RV2_size_features_ab.xls'
workbook.save(xls_file_path)

# image_path = 'subject101_frame01_slice01_label.png'
# white_pixel_count = count_size(image_path)
# print("白色像素点数:", white_pixel_count)