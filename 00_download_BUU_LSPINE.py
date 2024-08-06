import gdown

# Replace with your file ID
file_id = '1dfLSuBbbGyFTY8EoIppZfp4omtc8aSIc'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'BUU-LSPINE_V2.zip'  # Specify the output filename

gdown.download(url, output, quiet=False)
