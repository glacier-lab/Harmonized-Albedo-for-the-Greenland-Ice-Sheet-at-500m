cd /data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/src/polar/gportal
python3 generate_filelist.py \
    --start-date 2024-02-01 \
    --end-date 2026-11-24 \
    --chunk-days 7 \
    --bbox -180 60 180 90 \
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
