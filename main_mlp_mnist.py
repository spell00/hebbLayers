
def __main__():
    from MultiLayerPerceptron import MLP
    home_path = "/home/simon/"
    destination_folder = ""
    data_folder = "data"
    results_folder = "results"
    meta_destination_folder = "pandas_meta_df"
    plots_folder_path = "/".join([home_path, destination_folder, results_folder, "plots/"])

    dataset_name = "mnist_dropout"
    activation = "relu"
    early_stopping = 200
    n_epochs = 1000
    gt_input = 0
    lt_input = 3e6
    gt = -1e5
    extra_class = False
    use_conv = False  # Not applicable if not sequence (images, videos, sentences, DNA...)

    lr = 1e-4
    l1 = 0.
    l2 = 1e-6
    dropout = 0.5
    batch_size = 64
    is_pruning = True
    # mc = 1
    # iw = 1

    # Neurons layers
    h_dims = [1024, 1024]

    mlp = MLP(input_size=784, input_shape=(1, 28, 28), indices_names=list(range(784)),
              num_classes=10, h_dims=h_dims, extra_class=extra_class, l1=l1, l2=l2,
              gt_input=gt_input, lt_input=lt_input, gt=gt, is_pruning=is_pruning, dropout=dropout, labels_per_class=-1,
              destination_folder=home_path + "/" + destination_folder)

    mlp.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                    destination_folder=destination_folder, dataset_name=dataset_name, lr=lr,
                    meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers", num_classes=10, extra_class=True)

    mlp.load_example_dataset(dataset="mnist", batch_size=batch_size,
                             unlabelled_train_ds=False, normalize=True, mu=0.1307, var=0.3081,
                             unlabelled_samples=False)

    mlp.set_data(is_example=True, ignore_training_inputs=3, has_unlabelled_samples=False)

    mlp.cuda()
    # dgm.vae.generate_random(False, batch_size, z1_size, [1, 28, 28])
    mlp.run(n_epochs, hr=1, start_pruning=1, show_progress=2, ratio_replace=0.)

if __name__ == "__main__":
    __main__()
