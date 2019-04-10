#!/bin/bash

IDENTITY="IdentityFile=${AWS_KEY_PATH}"
sshfs -o $IDENTITY $AWS_USER@$AWS_HOST:/home/$AWS_USER /Volumes/git/sparsity-experiments/remote
