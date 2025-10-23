# Adobe-Internal


model			accuracy	loss			Parameters	inference(rtx 3060)

effNet_b3		0.9864		cross-entropy/BCELoss	13M		7.2 ms

Alexnet			0.9754		BCELoss			62M		2.7 ms

ViT-224-in21k(CNN Cat)	0.9901		cross-entropy		112M		7.8 ms

LeViT_128s		0.9784		cross-entropy		8M		2.4 ms

ViT-224-in21k		0.9831		cross-entropy		82M		6.3 ms

vgg16 (imagenet1k_v1)  	0.9787		cross-entropy		138M		9.90 ms
