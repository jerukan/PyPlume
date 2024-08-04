import click

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


if __name__ == "__main__":
    cli()
