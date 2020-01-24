Architecture like V1

Added Feature matching by breaking up the discriminator layer.

####more.py:
Used after epoch 181.
Trains the gen 4 times for each time it trains the discriminator.
The only issue I did the 3 times more training without the feature matching, which might cause mode collapse.

Maybe prone to Discriminator overfitting so I would want to add noise to the image. 
Also got some exploding gradients because the discriminator overfitting, with epochs with extreme changes:

epoch 648:
![](./readme_images/image_at_epoch_0648.png)

epoch 648:
![](./readme_images/image_at_epoch_0649.png)

#### even_more.py:
Got mode collapse after 695 epochs, with generator loss at around 4~5 and discriminator loss at around 0.2,

mode collapse evidence:
 
epoch 691:
![](./readme_images/image_at_epoch_0691.png)
So I started to train the gen while the loss is greater than X, but no more than 20 at a single batch.
Initialized X to 3.
The generator loss went to around 3.5 and disc loss went to around 0.2 at epoch 698.

After epoch 705 changed the generator learning rate to 1e-4 from 2e-4, to try to mitigate the exploding gradients.
This model is probably dead cause these methods did no good to reduce the model collapse and had issues due to the exploding gradients.

epoch 703:
![](./readme_images/image_at_epoch_0703.png)

epoch 704:
![](./readme_images/image_at_epoch_0704.png)