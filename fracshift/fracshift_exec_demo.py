from exec_centering import main

# parser.add_argument('sample', type=str)
# parser.add_argument('region_list', type=str)
# parser.add_argument('loop_path', type=str)
#
# parser.add_argument('-r', '--regions', type=str, default='all')
# parser.add_argument('-f', '--cooler_path', type=str, default='')
# parser.add_argument('-o', '--outfile', type=str, default='')
# parser.add_argument('-w', '--fracshift_window', type=int, default=5)
# parser.add_argument('-m', '--max_window', type=int, default=1)
# parser.add_argument('-l', '--logfile', type=str, default='logfile')
# parser.add_argument('-i', '--init_res', type=int, default=800)
# parser.add_argument('-t', '--target_res', type=int, default=200)
# parser.add_argument('-v', '--variable_res', action='store_true')

sample = 'GM12878'
region_list = '/mnt/md0/varshini/fracshift_demo/region_list.txt'
loop_path = '/mnt/md0/varshini/RCMC_LoopCaller/fracshift_demo/fracshift_demo_calls.txt'
regions = 'region1'
cooler_path = '/mnt/md0/varshini/RCMC_LoopCaller/data/GM12878_merged_realigned.50.mcool'
outfile = '/mnt/md0/varshini/RCMC_LoopCaller/loopcalls/fracshift_demo/demo.tsv'
logfile = '/mnt/md0/varshini/RCMC_LoopCaller/fracshift_demo/demo.log'

# test run with required arguments
main([sample, region_list, loop_path, cooler_path, outfile])

# test run with optional arguments
main([sample, region_list, loop_path, cooler_path, outfile, '-r', regions, '-l', logfile, '-v'])
