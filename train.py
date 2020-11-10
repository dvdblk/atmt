import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY


def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', default=False, help='Use a GPU')

    # Add data arguments
    parser.add_argument('--data', default='baseline/prepared_data', help='path to data directory')
    parser.add_argument('--source-lang', default='de', help='source language')
    parser.add_argument('--target-lang', default='en', help='target language')
    parser.add_argument('--max-tokens', default=None, type=int, help='maximum number of tokens in a batch')
    parser.add_argument('--batch-size', default=1, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--bpe-dropout', default=None, type=float, help='BPE dropout to apply for each training epoch')
    parser.add_argument('--train-on-tiny', action='store_true', help='train model on a tiny dataset')

    # Add model arguments
    parser.add_argument('--arch', default='lstm', choices=ARCH_MODEL_REGISTRY.keys(), help='model architecture')

    # Add optimization arguments
    parser.add_argument('--max-epoch', default=10000, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=4.0, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
    parser.add_argument('--adaptive-lr', action='store_true', help='whether an adaptive learning rate scheduler should be used')
    parser.add_argument('--patience', default=3, type=int,
                        help='number of epochs without improvement on validation set before early stopping')

    # Add checkpoint arguments
    parser.add_argument('--log-file', default=None, help='path to save logs')
    parser.add_argument('--save-dir', default='checkpoints', help='path to save checkpoints')
    parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')

    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(argument_default=argparse.SUPPRESS)
    ARCH_MODEL_REGISTRY[args.arch].add_args(model_parser)
    args = parser.parse_args()
    ARCH_CONFIG_REGISTRY[args.arch](args)
    return args

def bpe_dropout_if_needed(seed, bpe_dropout):
    """Apply BPE dropout to the training data if args contain bpe_dropout.
    
    Args:
        seed (int): the seed value to pass into apply_bpe --seed for reproducible dropout results
        bpe_dropout (float): the value that will be passed into apply_bpe --dropout
    """
    
    if bpe_dropout is None:
        # Dropout not needed
        return

    # Need to supply the full path to apply_bpe.py because using the symlink 
    # will ignore the -Wignore flag for some reason.
    ## DE
    preprocessed_data_prefix = os.path.join("model_bpe", "preprocessed_data")
    codes_fp = os.path.join(preprocessed_data_prefix, "bpe_codes")
    vocab_de = os.path.join(preprocessed_data_prefix, "vocab.de")
    vocab_en = os.path.join(preprocessed_data_prefix, "vocab.en")
    script_fp = os.path.join(os.path.join("subword_nmt", "subword_nmt"), "apply_bpe.py")
    bpe_de = os.path.join(preprocessed_data_prefix, "train.bpe.de")
    bpe_en = os.path.join(preprocessed_data_prefix, "train.bpe.en")
    os.system(
        'python -Wignore {} -c {} --vocabulary {} --vocabulary-threshold 1 --dropout {} --seed {} < {} > {}'.format(
           script_fp, codes_fp, vocab_de, bpe_dropout, seed, os.path.join(preprocessed_data_prefix, "train.de"), bpe_de
        )
    )
    ## EN
    os.system(
        'python -Wignore {} -c {} --vocabulary {} --vocabulary-threshold 1 --dropout {} --seed {} < {} > {}'.format(
            script_fp, codes_fp, vocab_en, bpe_dropout, seed, os.path.join(preprocessed_data_prefix, "train.en"), bpe_en
        )
    )

    # Preprocess only train
    prepared_data_prefix = os.path.join("model_bpe", "prepared_data")
    
    os.system(
        'python preprocess.py --target-lang en --source-lang de --dest-dir {} --vocab-src {} --vocab-trg {} --train-prefix {} --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000'.format(
            prepared_data_prefix, os.path.join(prepared_data_prefix, "dict.de"), os.path.join(prepared_data_prefix, "dict.en"), os.path.join(preprocessed_data_prefix, "train.bpe")
        )
    )
     
            

def main(args):
    """ Main training function. Trains the translation model over the course of several epochs, including dynamic
    learning rate adjustment and gradient clipping. """

    logging.info('Commencing training!')
    torch.manual_seed(42)

    utils.init_logging(args)

    # Load dictionaries
    src_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.source_lang)))
    logging.info('Loaded a source dictionary ({:s}) with {:d} words'.format(args.source_lang, len(src_dict)))
    tgt_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.target_lang)))
    logging.info('Loaded a target dictionary ({:s}) with {:d} words'.format(args.target_lang, len(tgt_dict)))

    # Load datasets
    def load_data(split):
        return Seq2SeqDataset(
            src_file=os.path.join(args.data, '{:s}.{:s}'.format(split, args.source_lang)),
            tgt_file=os.path.join(args.data, '{:s}.{:s}'.format(split, args.target_lang)),
            src_dict=src_dict, tgt_dict=tgt_dict)

    valid_dataset = load_data(split='valid')

    # Build model and optimization criterion
    model = models.build_model(args, src_dict, tgt_dict)
    logging.info('Built a model with {:d} parameters'.format(sum(p.numel() for p in model.parameters())))
    criterion = nn.CrossEntropyLoss(ignore_index=src_dict.pad_idx, reduction='sum')
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Instantiate optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)

    # Load last checkpoint if one exists
    state_dict = utils.load_checkpoint(args, model, optimizer, scheduler)  # lr_scheduler
    last_epoch = state_dict['last_epoch'] if state_dict is not None else -1

    # Track validation performance for early stopping
    bad_epochs = 0
    best_validate = float('inf')

    for epoch in range(last_epoch + 1, args.max_epoch):
        ## BPE Dropout
        # Set the seed to be equal to the epoch 
        # (this way we guarantee same seeds over multiple training runs, but not for each training epoch)
        seed = epoch

        bpe_dropout_if_needed(seed, args.bpe_dropout)

        # Load the BPE (dropout-ed) training data
        train_dataset = load_data(split='train') if not args.train_on_tiny else load_data(split='tiny_train')
        train_loader = \
            torch.utils.data.DataLoader(train_dataset, num_workers=1, collate_fn=train_dataset.collater,
                                        batch_sampler=BatchSampler(train_dataset, args.max_tokens, args.batch_size, 1,
                                                                   0, shuffle=True, seed=42))
        model.train()
        stats = OrderedDict()
        stats['loss'] = 0
        stats['lr'] = 0
        stats['num_tokens'] = 0
        stats['batch_size'] = 0
        stats['grad_norm'] = 0
        stats['clip'] = 0
        # Display progress
        progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)

        # Iterate over the training set
        for i, sample in enumerate(progress_bar):
            if args.cuda:
                sample = utils.move_to_cuda(sample)
            if len(sample) == 0:
                continue
            model.train()

            output, _ = model(sample['src_tokens'], sample['src_lengths'], sample['tgt_inputs'])
            loss = \
                criterion(output.view(-1, output.size(-1)), sample['tgt_tokens'].view(-1)) / len(sample['src_lengths'])
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            optimizer.zero_grad()

            # Update statistics for progress bar
            total_loss, num_tokens, batch_size = loss.item(), sample['num_tokens'], len(sample['src_tokens'])
            stats['loss'] += total_loss * len(sample['src_lengths']) / sample['num_tokens']
            stats['lr'] += optimizer.param_groups[0]['lr']
            stats['num_tokens'] += num_tokens / len(sample['src_tokens'])
            stats['batch_size'] += batch_size
            stats['grad_norm'] += grad_norm
            stats['clip'] += 1 if grad_norm > args.clip_norm else 0
            progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
                                     refresh=True)

        logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
            value / len(progress_bar)) for key, value in stats.items())))

        # Calculate validation loss
        valid_perplexity, valid_loss = validate(args, model, criterion, valid_dataset, epoch)
        model.train()

        # Scheduler step
        if args.adaptive_lr:
            scheduler.step(valid_loss)

        # Save checkpoints
        if epoch % args.save_interval == 0:
            utils.save_checkpoint(args, model, optimizer, scheduler, epoch, valid_perplexity)  # lr_scheduler

        # Check whether to terminate training
        if valid_perplexity < best_validate:
            best_validate = valid_perplexity
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            logging.info('No validation set improvements observed for {:d} epochs. Early stop!'.format(args.patience))
            break


def validate(args, model, criterion, valid_dataset, epoch):
    """ Validates model performance on a held-out development set. """
    valid_loader = \
        torch.utils.data.DataLoader(valid_dataset, num_workers=1, collate_fn=valid_dataset.collater,
                                    batch_sampler=BatchSampler(valid_dataset, args.max_tokens, args.batch_size, 1, 0,
                                                               shuffle=False, seed=42))
    model.eval()
    stats = OrderedDict()
    stats['valid_loss'] = 0
    stats['num_tokens'] = 0
    stats['batch_size'] = 0

    # Iterate over the validation set
    for i, sample in enumerate(valid_loader):
        if args.cuda:
            sample = utils.move_to_cuda(sample)
        if len(sample) == 0:
            continue
        with torch.no_grad():
            # Compute loss
            output, attn_scores = model(sample['src_tokens'], sample['src_lengths'], sample['tgt_inputs'])
            loss = criterion(output.view(-1, output.size(-1)), sample['tgt_tokens'].view(-1))
        # Update tracked statistics
        stats['valid_loss'] += loss.item()
        stats['num_tokens'] += sample['num_tokens']
        stats['batch_size'] += len(sample['src_tokens'])

    val_loss = stats['valid_loss']

    # Calculate validation perplexity
    stats['valid_loss'] = stats['valid_loss'] / stats['num_tokens']
    perplexity = np.exp(stats['valid_loss'])
    stats['num_tokens'] = stats['num_tokens'] / stats['batch_size']

    logging.info(
        'Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(value) for key, value in stats.items())) +
        ' | valid_perplexity {:.3g}'.format(perplexity))

    return perplexity, val_loss


if __name__ == '__main__':
    args = get_args()
    args.device_id = 0

    # Set up logging to file
    logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    if args.log_file is not None:
        # Logging to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    main(args)
