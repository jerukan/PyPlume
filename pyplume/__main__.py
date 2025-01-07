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
        savedir = Path(reginfo["folder"])
        savedir.mkdir(exist_ok=True, parents=True)
        url = reginfo["url"]
        infoargs = {
            "time_range": reginfo["time_range"],
            "lat_range": reginfo["lat_range"],
            "lon_range": reginfo["lon_range"],
            "inclusive": reginfo["inclusive"]
        }
        with dataloaders.DataLoader(url, **infoargs) as dl:
            megabytes = dl.dataset.nbytes / 1e6
            print(f"Downloading {regname} ({megabytes:.2f} MB)...")
            savepath = savedir / f"{regname}.nc"
            dl.save(savepath)
            print(f"Saved to {savepath}")


if __name__ == "__main__":
    cli()
