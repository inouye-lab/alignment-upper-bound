from subprocess import PIPE, run, call, check_call, Popen
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

def out(command):
    result = run([command], stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)

    # result = Popen(command, stdout=PIPE, stderr=PIPE)
    # output, error = result.communicate()
    # print('error',error)
    # return output
    return result.stdout

def main(args):

    f = open(os.path.join(args.result_dir, f"FID-for-{args.target_numbers}.txt"), "w")

    for trg in args.target_numbers:
        for src in args.target_numbers:
            if trg != src:
                real_dir = os.path.join(args.real_image_dir, "MNIST_dataset", "test", trg)
                fake_dir = os.path.join(args.fake_image_dir, f"{src}to{trg}")  # 1to0
                my_output = out(f"python -m pytorch_fid {real_dir} {fake_dir}  --device {args.device}")  # real 0 and fake 1to0
                # my_output = out(["/apps/spack/gilbreth/apps/anaconda/2020.11-py38-gcc-4.8.5-djkvkvk/bin/python", "-m", "pytorch_fid", f"{real_dir}", f"{fake_dir}", "--device", f"{args.device}"])  # real 0 and fake 1to0
                # my_output = my_output.decode("utf-8")
                print(f"{src} to {trg}", my_output)
                f.write(f"FID between real {trg} and fake {trg} from {src} is: {my_output}\n")

if __name__ == '__main__':
    print('get started')
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--target_numbers', type=str, default='012',
                        help='which classes is using')
    parser.add_argument('--real_image_dir', type=str, default="/scratch/gilbreth/cho436/data",
                        help='where is the real images located')
    # parser.add_argument('--fake_image_dir', type=str,
    #                     default="/scratch/gilbreth/cho436/experiment_results/012_AlignFlow",
    #                     help='where is the fake images located')
    parser.add_argument('--fake_image_dir', type=str,
                        default="/scratch/gilbreth/cho436/experiment_results",
                        help='where is the fake images located')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use. Like cuda, cuda:0 or cpu')
    # parser.add_argument('--result_dir', type=str, default="/scratch/gilbreth/cho436/experiment_results/012_AlignFlow",
    #                     help=('Paths to save the FID score'))
    parser.add_argument('--result_dir', type=str, default="/scratch/gilbreth/cho436/experiment_results",
                        help=('Paths to save the FID score'))

    parser.add_argument('--setting', type=str, default="012_ours_TC",
                        help=('Paths to save the FID score'))

    args = parser.parse_args()
    args.fake_image_dir = os.path.join(args.fake_image_dir, args.setting)
    args.result_dir = os.path.join(args.result_dir, args.setting)
    main(args)
