import torch
from load import load_acm, load_mag, load_freebase, load_cs, load_Aminer_Large, load_imdb, load_cite, load_nn, load_oag
from params import set_params, acm_params, aminer_params, freebase_params, cs_params, imdb_params, cite_params, \
    mag_params
from evaluate import evaluate
from model import CG
from torch_geometric import seed_everything
import warnings

warnings.filterwarnings("ignore")

# args = acm_params()
# args = aminer_params()
# args = cite_params()
# args = imdb_params()
# args = freebase_params()
args = cs_params()
# args = mag_params()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if args.dataset == "acm":
    load_data = load_acm
elif args.dataset == "aminer":
    load_data = load_Aminer_Large
elif args.dataset == "imdb":
    load_data = load_imdb
elif args.dataset == "cite":
    load_data = load_cite
elif args.dataset == "freebase":
    load_data = load_freebase
elif args.dataset == "cs":
    load_data = load_oag
elif args.dataset == "mag":
    load_data = load_mag

data = load_data()
# label = data[data.main_node].y.cpu().long()
# del data[data.main_node].y
data = data.to(device)

print(data)


def main(space):
    seed_everything(0)

    model = CG(data=data, hidden_dim=64, feat_drop=space['feat'],
               att_drop1=space['attr1'], att_drop2=space['attr2'], r1=0.1,
               r2=0.1, r3=0.1, b=space['b']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=space['lr'], weight_decay=space['w'])
    for epoch in range(1, int(space['epoch']) + 1):
        model.train()
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()

    model.eval()
    embeds = model.get_embed(data).cpu()

    f1 = evaluate(
        embeds,
        data[data.main_node].train.cpu(),
        data[data.main_node].val.cpu(),
        data[data.main_node].test.cpu(),
        data[data.main_node].y.cpu().long(),
        # label,
        torch.device("cuda:0"),
        data,
        0.01,
        0,
        args.dataset
    )
    space['acc'] = f1
    print(space)


if __name__ == '__main__':
    # Mater
    # main({'attr1': 0.8, 'attr2': 0.8, 'b': 2.0, 'epoch': 3500, 'feat': 0.0, 'lr': 0.0001, 'w': 0.0005, 'acc': 0.51679})
    # Field
    # main({'attr1': 0.3, 'attr2': 0.9, 'b': 1.9, 'epoch': 8000, 'feat': 0.3, 'lr': 8e-5, 'w': 0.001, 'acc': 0.4916})
    # ML
    # main({'attr1': 0.7, 'attr2': 0.1, 'b': 1.9, 'epoch': 3100, 'feat': 0.1, 'lr': 0.0005, 'w': 0.0001, 'acc': 0.4008})
    # Field
    # main({'attr1': 0.6, 'attr2': 0.1, 'b': 1.8, 'epoch': 5000, 'feat': 0.2, 'lr': 0.0008, 'w': 1e-5, 'acc': 0.32706})
    # NN
    # main({'attr1': 0.0, 'attr2': 0.5, 'b': 1.2, 'epoch': 8000, 'feat': 0.3, 'lr': 0.008, 'w': 1e-05, 'acc': 0.3476})
    # Field
    # main({'attr1': 0.5, 'attr2': 0.0, 'b': 1.5, 'epoch': 2000, 'feat': 0.0, 'lr': 0.008, 'w': 0.0001, 'acc': 0.50378})
    # Business
    main({'attr1': 0.6, 'attr2': 0.8, 'b': 0.7, 'epoch': 2400, 'feat': 0.5, 'lr': 0.0005, 'w': 0.0001, 'acc': 0.4562})
    # Field
    # main({'attr1': 0.6, 'attr2': 0.8, 'b': 0.2, 'epoch': 10000, 'feat': 0.7, 'lr': 0.0005, 'w': 1e-5, 'acc': 0.317})
    # Art
    # main({'attr1': 0.7, 'attr2': 0.8, 'b': 0.2, 'epoch': 3000, 'feat': 0.1, 'lr': 0.0008, 'w': 1e-5, 'acc': 0.2856})
    # Field
    # main({'attr1': 0.6, 'attr2': 0.8, 'b': 0.3, 'epoch': 5000, 'feat': 0.2, 'lr': 0.0008, 'w': 5e-5, 'acc': 0.3680})
    # Engin
    # main({'attr1': 0.2, 'attr2': 0.8, 'b': 1.8, 'epoch': 10000, 'feat': 0.5, 'lr': 1e-5, 'w': 5e-5, 'acc': 0.5107})
    # Field
    # main({'attr1': 0.2, 'attr2': 0.6, 'b': 1.5, 'epoch': 10000, 'feat': 0.2, 'lr': 0.0005, 'w': 0.0001, 'acc': 0.3903})
    # CS
    # main({'attr1': 0.2, 'attr2': 0.6, 'b': 1.4, 'epoch': 6500, 'feat': 0.7, 'lr': 0.0005, 'w': 5e-5, 'acc': 0.42636})
    # Chem
    # main({'attr1': 0.3, 'attr2': 0.8, 'b': 1.8, 'epoch': 6000, 'feat': 0.6, 'lr': 0.001, 'w': 1e-5, 'acc': 0.4847})
