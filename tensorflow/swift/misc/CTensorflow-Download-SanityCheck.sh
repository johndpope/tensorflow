TF_TYPE="cpu" # Change to "gpu" for GPU support
 OS="darwin" # Change to "darwin" for Mac OS
 TARGET_DIRECTORY="/usr/local"
 curl -L \
   "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-${OS}-x86_64-1.1.0.tar.gz" |
   sudo tar -C $TARGET_DIRECTORY -xz


   gcc -I/usr/local/include -L/usr/local/lib hello_tf.c -ltensorflow -o helloTensorflow