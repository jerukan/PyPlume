import click

from pyplume.config_parser import load_config, prep_sim_from_cfg, process_results


@click.group()
def cli():
    pass


@cli.command()
@click.option("-c", "--cfg", "config_path", type=click.Path(exists=True))
def simulate(config_path):
    loaded_config = load_config(config_path)
    sim = prep_sim_from_cfg(loaded_config)
    sim.execute()
    process_results(sim, loaded_config)


if __name__ == "__main__":
    cli()
