from pathlib import Path

import click
import yaml

from pyplume import dataloaders
from pyplume.config_parser import load_config, prep_sims_from_cfg, process_results


@click.group()
def cli():
    pass


@cli.command()
@click.option("-c", "--cfg", "config_path", type=click.Path(exists=True, dir_okay=False))
def simulate(config_path):
    loaded_config = load_config(config_path)
    sims = prep_sims_from_cfg(loaded_config)
    for sim in sims:
        sim.execute()
        process_results(sim, loaded_config)


@cli.command()
@click.option("-c", "--cfg", "config_path", type=click.Path(exists=True, dir_okay=False))
def downloadnc(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for regname, reginfo in config.items():
        savedir = Path(reginfo.pop("folder"))
        savedir.mkdir(exist_ok=True, parents=True)
        url = reginfo.pop("url")
        filesplit = reginfo.pop("filesplit", 0)
        overwrite = reginfo.pop("overwrite", True)
        reginfo["load_into_memory"] = False
        with dataloaders.DataLoader(url, **reginfo) as dl:
            megabytes = dl.dataset.nbytes / 1e6
            print(f"Downloading {regname} ({megabytes:.2f} MB)...")
            if filesplit > 0:
                savepath = savedir / f"{regname}"
                dl.save(savepath, filesplit=filesplit, overwrite=overwrite)
            else:
                savepath = savedir / f"{regname}.nc"
                dl.save(savepath, filesplit=0, overwrite=overwrite)
            print(f"Saved to {savepath}")


if __name__ == "__main__":
    cli()
