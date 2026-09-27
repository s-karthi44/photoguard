"""
=========================================================
PhotoGuard AI
CNN Trainer

Author : Vinay

Description:
Contains reusable training and validation loops.

=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import torch
from tqdm import tqdm

from modules.cnn.config import DEVICE
from modules.cnn.utils import AverageMeter


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
):

    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress_bar:

        labels = labels.to(DEVICE , non_blocking = True)
        images = images.to(DEVICE , non_blocking = True)

        # -------------------------------
        # Clear Previous Gradients
        # -------------------------------

        optimizer.zero_grad()

        # -------------------------------
        # Forward Pass
        # -------------------------------

        outputs = model(images)

        loss = criterion(outputs, labels)

        # -------------------------------
        # Backpropagation
        # -------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(

        model.parameters(),

         max_norm=1.0

)

        optimizer.step()

        # -------------------------------
        # Calculate Accuracy
        # -------------------------------

        predicted = outputs.argmax(dim=1)

        correct = (predicted == labels).sum().item()

        accuracy = correct / labels.size(0)

        # -------------------------------
        # Update Meters
        # -------------------------------

        loss_meter.update(
            loss.item(),
            images.size(0)
        )

        accuracy_meter.update(
            accuracy,
            images.size(0)
        )

        # -------------------------------
        # Progress Bar
        # -------------------------------

        progress_bar.set_postfix({

            "Loss": f"{loss_meter.avg:.4f}",

            "Accuracy": f"{accuracy_meter.avg*100:.2f}%"

        })

    return (

        loss_meter.avg,

        accuracy_meter.avg

    )
# =========================================================
# VALIDATE ONE EPOCH
# =========================================================

def validate_one_epoch(
    model,
    validation_loader,
    criterion,
):

    model.eval()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    progress_bar = tqdm(
        validation_loader,
        desc="Validation",
        leave=False
    )

    with torch.no_grad():

        for images, labels in progress_bar:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # ---------------------------------
            # Forward Pass
            # ---------------------------------

            with torch.autocast(

                device_type=DEVICE.type,

                enabled=False
            ):

                outputs = model(images)

                loss = criterion(outputs, labels)
            # ---------------------------------
            # Accuracy
            # ---------------------------------

            predicted = outputs.argmax(dim=1)

            correct = (predicted == labels).sum().item()

            accuracy = correct / labels.size(0)

            # ---------------------------------
            # Update Metrics
            # ---------------------------------

            loss_meter.update(
                loss.item(),
                images.size(0)
            )

            accuracy_meter.update(
                accuracy,
                images.size(0)
            )

            progress_bar.set_postfix({

             "Loss": f"{loss_meter.avg:.4f}",

             "Accuracy": f"{accuracy_meter.avg*100:.2f}%"

            })

    return (

        loss_meter.avg,

        accuracy_meter.avg

    )


# =========================================================
# TEST TRAINER
# =========================================================

def trainer_info():

    print("\n" + "=" * 60)

    print("PhotoGuard AI")

    print("CNN Trainer Ready")

    print("=" * 60)


# =========================================================
# MAIN
# =========================================================

def main():

    trainer_info()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()