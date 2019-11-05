hr = imread(char("./results1/video13_536x960_60fps_10_x1_HR.png"));
sr = imread(char("./results1/video13_536x960_60fps_10_x1_SR.png"));
lr = imread(char("./results1/video13_536x960_60fps_10_x1_lR.png"));
x1 = abs(hr-sr);
x2 = abs(hr-lr);