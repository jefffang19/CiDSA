from train import training
from eval import evaluation

#  You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        default='eval',
                        help='input eval or train')

    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input eletricity training data 1 file name')

    parser.add_argument('--training2',
                        help='input eletricity training data 2 file name')

    parser.add_argument('--weather_past',
                        help='input weather training data 1 (past) file name')

    parser.add_argument('--weather_forecast',
                        help='input weather training data 2 (forecast) file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    if args.mode == 'train':
        print('training mode')
        training(args.training, args.training2,
                 args.weather_past, args.weather_forecast, args.output, epoches=100)
    elif args.mode == 'eval':
        print('eval mode')
        evaluation(args.training, args.training2,
                   args.weather_past, args.weather_forecast, args.output)

    else:
        print('unknown mode, please try \'train\' or \'eval\'')
