configs = {
    "S5_pendulum": {
        "run_name": 'S5,8,64,64,4,t[0-50:80],rng',
        "load_run": None,
        "jax_seed": 69,
        "lr_finder": False,
        "warmup_steps": 1000,
        "training_steps": 9000,
        "batch_size": 64,
        "learning_rate": 3e-3,
        "weight_decay": 0.05,
        "architecture": 'S5',
        "dense_params": {
            "hidden_dim": 128,
            "out_dim": 2,
            "dropout": 0.1},
        "cnn_params": {
            "dense_dim": 256,
            "out_dim": 2,
            "dropout": 0.},
        "s5_params": {
            "n_layers": 8,
            "d_model": 64,
            "ssm_size": 64,
            "blocks": 4,
            "out_dim": 30,
            "activation": 'half_glu2',
            "prenorm": False,
            "batchnorm": False,
            "dropout": 0.1}
    },
    "Crypto-v0": {
        # Add hyperparameters for Crypto dataset here
    }
}