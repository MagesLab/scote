Configuration of the workflow.

- `datasets.yaml`: URL, archive name, size, license, archive members and platform per data set; entries with
  `manual_download: true` point to a vendor landing page and expect the file in place.
- `host.yaml`: the host overlay, i.e. the free-space floor of the data volume and the directory for the benchmark
  outputs of Fig. 6. Pass another host file with `--configfile config/datasets.yaml my_host.yaml` to override it.
- `operators_spleen.yaml`: the molecule operators and replicate groups of Fig. 3.

All data go to `data/` in the examples root (about 1 TB); make `data/` a link to a large volume.
