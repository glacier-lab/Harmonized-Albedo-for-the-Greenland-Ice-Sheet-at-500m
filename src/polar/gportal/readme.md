cd /data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/src/polar/gportal
python3 generate_filelist.py \
    --start-date 2018-01-01 \
    --end-date 2026-11-24 \
    --chunk-days 7 \
    --bbox -180 60 180 90 \
    --overwrite

python3 generate_filelist.py \
    --start-date 2018-01-01 \
    --end-date 2026-11-24 \
    --chunk-days 7 \
    --bbox -180 -90 180 -60 \
    --overwrite

`generate_filelist.py` splits the requested period into inclusive date windows,
queries all CSW result pages in each window, and appends new, unique `get`
commands to `files.lst`. Existing entries are retained by default, including
when a run starts again after an earlier run or interruption. Use `--overwrite`
to start with an empty list.

Run `python3 generate_filelist.py --help` for options such as `--dataset-id`,
`--output`, `--count`, and `--insecure`. The page count cannot exceed G-Portal's
3,000-record limit.

cp /data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/src/polar/gportal/files.lst /data_3/shunan_2/AU/hsa500m/Arctic/GCOM-C-SR/
cd /data_3/shunan_2/AU/hsa500m/Arctic/GCOM-C-SR
./sftp_get.sh

Alternatively, you can use the `download_filelist.py` script to download the files.
To download the generated list:

```bash
python3 download_filelist.py \
    --batch-file files.lst \
    --destination /data_3/shunan_2/AU/hsa500m/Arctic/GCOM-C-SR \
    --workers 2
```

The downloader opens one persistent SFTP connection per worker. It skips files
already present in the destination and uses `.part` files so incomplete
transfers are not mistaken for completed products. Set `GPORTAL_USERNAME` and
`GPORTAL_PRIVATE_KEY`, or pass `--username` and `--private-key`, to override the
configured defaults.

Parallel downloads can be faster for many files, high-latency transfers, or
per-connection throttling. If G-Portal limits aggregate account bandwidth, more
workers will not increase total throughput and may add server load. Start with 2
workers and benchmark a representative subset using `--workers 1`,
`--workers 2`, and `--workers 4`; each run reports elapsed time.

Use `--dry-run` to validate the batch and list pending files without connecting.
