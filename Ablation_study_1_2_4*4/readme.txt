To reproduce the result, need to download the categorywise ( training, validation and testing) dataset from: https://drive.google.com/drive/folders/1Okq7vNKhzJA8dElRKqcIYpblTyxpDWes?usp=sharing. After downloading the dataset, just use the local folder location in the train_ablation.py file. Need to change the directory address ( default=r"..../ANONYMIZED_Denoise_Split_Train_Valid_Test_Data) from code. 

                       (parser.add_argument('--data_dir', type=str, 
                        default=r"..../ANONYMIZED_Denoise_Split_Train_Valid_Test_Data",
                        help='Root directory containing traindata/ and validationdata/'))
