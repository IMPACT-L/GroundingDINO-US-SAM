trainer = GroundingDINOTrainer(
        model,
        num_steps_per_epoch=steps_per_epoch,
        num_epochs=training_config.num_epochs,
        warmup_epochs=training_config.warmup_epochs,
        learning_rate=training_config.learning_rate,
        use_lora=training_config.use_lora
    )   