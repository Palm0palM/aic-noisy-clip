for p in ['outputs/prelim_run2_352real/pred_raw_352real.csv',
          'outputs/prelim_run2_352real/pred_r352_T6_t16.csv']:
    print('===', p)
    with open(p) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(repr(line))
