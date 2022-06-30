from mislight.options import PreprocessOptions as Options
from mislight.engine.preprocessing import MyPreprocessing

if __name__ == '__main__':
    opt = Options().parse()
    
    pp = MyPreprocessing(opt)
    if pp.check_data(opt.check_resample_target):
        pp.resample_data()
        pp.save_dataset()
