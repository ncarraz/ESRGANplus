import matlab.engine
# addpath('matlab/myfiles','-end')
eng = matlab.engine.start_matlab()
eng.addpath(r'/Users/broz/Documents/MATLAB/blind_image_quality_toolbox',nargout=0)
eng.addpath(r'/Users/broz/Documents/MATLAB/sr-metric',nargout=0)
eng.score_image(nargout=0)
eng.quit()

