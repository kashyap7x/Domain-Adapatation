import config as cfg
import office_model
import utils


if __name__ == '__main__':
    # Initialize model and trainer
    model = office_model.office_model(cfg.init_lr, cfg.synth_lr, cfg.momentum, cfg.weight_decay)
    trainer = office_model.synthetic_trainer(model, cfg.phases, cfg.alpha, cfg.beta, cfg.tau)

    # Train
    model_best = trainer.train_model(cfg.ratios, cfg.batch_size, cfg.dset_sizes, cfg.dset_loaders,
                                     utils.inv_lr_scheduler, cfg.init_lr, cfg.gamma, cfg.power,
                                     gpu_id=cfg.gpu_id, save_best=cfg.save_best, maxIter=cfg.maxIter)