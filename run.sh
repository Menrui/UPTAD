# python main.py mode=run dataset=yamaha model=ae model.nch=64 model.loss_type=L2,L1 -m
# python main.py mode=run dataset=yamaha model=ae model.nch=128 model.loss_type=L2,L1 -m
# python main.py mode=run dataset=yamaha model=ae model.nch=256 model.loss_type=L2,L1 -m
# python main.py mode=run dataset=yamaha model=ae model.nch=512 model.loss_type=L2,L1 dataset.batch_size=32 -m
# python main.py mode=run dataset=yamaha model=ae model.nch=1024 model.loss_type=L2,L1 dataset.batch_size=32 -m

# python main.py mode=run dataset=wood model=ae model.loss_type=Texture


# python main.py mode=run dataset=yamaha model=ae model.nch=128 model.loss_type=L1
# python main.py mode=run dataset=yamaha model=ae model.nch=256 model.loss_type=L1
# python main.py mode=run dataset=yamaha model=ae model.nch=512 model.loss_type=L1

# python main.py mode=run dataset=wood model=ae model.loss_type=Texture


# python main.py mode=run model=unet dataset=wood model.mask.add_mask=True model.mask.add_loss_mask=True model.loss_type=Texture \
#     model.mask.mask_w=3 model.mask.mask_h=3 mode.num_epochs=300 seed=999,998,997 -m
# python main.py mode=run model=unet dataset=wood model.mask.add_mask=True model.mask.add_loss_mask=False model.loss_type=Texture \
#     model.mask.mask_w=3 model.mask.mask_h=3 mode.num_epochs=300 seed=999,998,997 -m
# python main.py mode=run model=unet dataset=wood model.mask.add_mask=True model.mask.add_loss_mask=True model.loss_type=Texture \
#     model.mask.mask_w=5 model.mask.mask_h=5 mode.num_epochs=300 seed=999,998,997 -m
# python main.py mode=run model=unet dataset=wood model.mask.add_mask=True model.mask.add_loss_mask=True model.loss_type=Texture \
#     model.mask.mask_w=7 model.mask.mask_h=7 mode.num_epochs=300 seed=999,998,997 -m
# python main.py mode=run model=unet dataset=wood model.mask.add_mask=True model.mask.add_loss_mask=True model.loss_type=Texture \
#     model.mask.mask_w=9 model.mask.mask_h=9 mode.num_epochs=300 seed=999,998,997 -m


# python main.py mode=run model=unet dataset=mvtec model.mask.add_mask=True model.mask.add_loss_mask=True model.loss_type=Texture \
#     model.mask.mask_w=5 model.mask.mask_h=5 mode.num_epochs=300 seed=999 mode.score_type=L1max,L2max,SSIM,SP,L1SP -m
# python main.py mode=run model=ae dataset=mvtec model.mask.add_mask=False model.loss_type=Texture \
#     mode.num_epochs=300 seed=999 mode.score_type=L1max,L2max,SSIM,SP,L1SP -m

# python main.py mode=run model=unet dataset=mvtec model.mask.add_mask=True model.mask.add_loss_mask=True model.loss_type=Texture \
#     model.mask.mask_w=5 model.mask.mask_h=5 mode.num_epochs=300 seed=999 mode.score_type=PatchSSIM mode.score.style_alpha=10
# python main.py mode=run model=ae dataset=mvtec model.mask.add_mask=False model.loss_type=Texture \
#     mode.num_epochs=300 seed=999 mode.score_type=PatchSSIM
# python main.py mode=run model=unet dataset=mvtec model.mask.add_mask=True model.mask.add_loss_mask=True model.loss_type=Texture \
#     model.mask.mask_w=5 model.mask.mask_h=5 mode.num_epochs=300 seed=999 mode.score_type=PatchSSIM mode.score.style_alpha=120
# python main.py mode=run model=unet dataset=mvtec model.mask.add_mask=True model.mask.add_loss_mask=True model.loss_type=Texture \
#     model.mask.mask_w=3 model.mask.mask_h=3 mode.num_epochs=300 seed=999 mode.score_type=PatchSSIM mode.score.style_alpha=120
# python main.py mode=run model=unet dataset=mvtec model.mask.add_mask=True model.mask.add_loss_mask=True model.loss_type=L1 \
#     model.mask.mask_w=3 model.mask.mask_h=3 mode.num_epochs=300 seed=999 mode.score_type=PatchSSIM mode.score.style_alpha=120

# python main.py mode=run model=unet dataset=mvtec model.mask.add_mask=True model.mask.add_loss_mask=True model.loss_type=Texture \
#     model.mask.mask_w=5 model.mask.mask_h=5 mode.num_epochs=301 seed=999 mode.score_type=PatchSSIM mode.score.style_alpha=120 mode.score.perceptual_alpha=1,0.5,0.05 mode.score.l1_alpha=5 mode.score.smooth_alpha=1,0 -m

# python main.py mode=run model=unet dataset=mvtec model.mask.add_mask=True model.mask.add_loss_mask=True model.loss_type=SSIM \
#     model.mask.mask_w=5 model.mask.mask_h=5 mode.num_epochs=302 seed=999 mode.score_type=PatchSSIM
# python main.py mode=run model=unet dataset=mvtec model.mask.add_mask=True model.mask.add_loss_mask=True model.loss_type=SSIM \
#     model.mask.mask_w=3 model.mask.mask_h=3 mode.num_epochs=302 seed=999 mode.score_type=PatchSSIM

# python main.py mode=run model=unet dataset=mvtec model.mask.add_mask=True model.mask.add_loss_mask=True model.loss_type=SSIM \
#     model.mask.mask_w=3 model.mask.mask_h=3 mode.num_epochs=50 seed=999 mode.score_type=PatchSSIM,L1max,L2max -m

python main.py mode=run model=unet dataset=mvtec model.mask.add_mask=True model.mask.add_loss_mask=False model.loss_type=Texture \
    model.mask.mask_w=5 model.mask.mask_h=5 mode.num_epochs=151 seed=999 mode.score_type=PatchSSIM,L1max,L2max -m


# python main.py mode=run model=ae dataset=yamaha dataset.category=color_pattert_d5_crop_v3 model.loss_type=SSIM,L1,L2 mode.score_type=PatchSSIM,L1max,L2max mode.num_epochs=301 -m