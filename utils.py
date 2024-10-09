import pickle

def save_model(
    cls_model_path:str ='cls_model.pkl' , 
    pca_model_path:str ='pca_model.pkl' ,
    model:object = None,
    pca:object = None
):
    # save
    with open(cls_model_path,'wb') as f:
        pickle.dump(model,f)

    with open(pca_model_path,'wb') as f:
        pickle.dump(pca,f)

def save_data(txt_output_file, file_list):
    with open(txt_output_file, 'w') as f:
        for file in file_list:
            f.write(file + '\n')