unzip $HOME/.keras/datasets/cats-dogs.zip.tar.gz -d $HOME/.keras/datasets
mv $HOME/.keras/datasets/PetImages $HOME/.keras/datasets/cat_set
rm $HOME/.keras/datasets/MSR-LA\ -\ 3467.docx
rm $HOME/.keras/datasets/readme\[1\].txt 
mv $HOME/.keras/datasets/cat_set/Cat $HOME/.keras/datasets/cat_set/cat
mv $HOME/.keras/datasets/cat_set/Dog $HOME/.keras/datasets/cat_set/dog

#pip3 install image
#pip3 install python-dateutil
#there is 666.jpg in cats dataset it MUST be removed, cause one crashes script
#interesting what is the limit?
