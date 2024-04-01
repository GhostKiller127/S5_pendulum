configs = {
    "S5_pendulum": {
        "run_name": 'S5,8,64,64,4,t[l=40],v[60:100]',
        "load_run": None,
        "jax_seed": 69,
        "lr_finder": False,
        "warmup_steps": 1000,
        "training_steps": 9000,
        "batch_size": 64,
        "learning_rate": 3e-3,
        "weight_decay": 0.05,
        "architecture": 'S5',
        "parameters": {
            "Dense": {
                "input_shape": (1, 576, 1),
                "hidden_dim": 128,
                "out_dim": 2,
                "dropout": 0.1,
                },
            "CNN": {
                "input_shape": (1, 24, 24, 1),
                "dense_dim": 256,
                "out_dim": 2,
                "dropout": 0.,
                },
            "S5": {
                # "input_shape": (1, 100, 24, 24, 1),
                "input_shape": (32, 10000, 25),
                "n_layers": 4,
                "d_model": 32,
                "ssm_size": 32,
                "blocks": 4,
                "decoder_dim": 30,
                "output_dim": 10,
                "activation": 'half_glu2',
                "prenorm": False,
                "batchnorm": False,
                "dropout": 0.1,
            },
        },
    },
    "Crypto": {
        # Add hyperparameters for Crypto task here
    }
}