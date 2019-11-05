fdir='./result\';
% f=fullfile(fdir,'BasketballDrive*_HR.png');
% dirOutput=dir(f);
% filenames={dirOutput.name}';
% length = size(filenames,1);
% s1=0;
% s2=0;
% list1=[];
% list2=[];
% for i = 1: length
%     hr=imread(char(strcat(fdir, filenames(i))));
%     sr=imread(char(strcat(fdir,strrep(filenames(i),'_HR','_SR'))));
%     lr=imread(char(strcat(fdir,strrep(filenames(i),'_HR','_LR'))));
%     psnr_h_s=psnr(hr,sr);
%     psnr_h_l=psnr(hr,lr);
%     %fprintf('%d,%s,psnr_h_s=,%f,psnr_h_l=%f\n',i,char(filenames(i)),psnr_h_s,psnr_h_l);
%     s1 = s1+psnr_h_s;
%     s2 = s2+psnr_h_l;
%     list1=[list1,psnr_h_s];
%     list2=[list2,psnr_h_l];
% end
% s11=s1/length;
% s12=s2/length;
% f=fullfile(fdir,'/pedestrianarea*_HR.png');
% dirOutput=dir(f);
% filenames={dirOutput.name}';
% length = size(filenames,1);
% s1=0;
% s2=0;
% list1=[];
% list2=[];
% for i = 1: length
%     hr=imread(char(strcat(fdir, filenames(i))));
%     sr=imread(char(strcat(fdir,strrep(filenames(i),'_HR','_SR'))));
%     lr=imread(char(strcat(fdir,strrep(filenames(i),'_HR','_LR'))));
%     psnr_h_s=psnr(hr,sr);
%     psnr_h_l=psnr(hr,lr);
%     %fprintf('%d,%s,psnr_h_s=,%f,psnr_h_l=%f\n',i,char(filenames(i)),psnr_h_s,psnr_h_l);
%     s1 = s1+psnr_h_s;
%     s2 = s2+psnr_h_l;
%     list1=[list1,psnr_h_s];
%     list2=[list2,psnr_h_l];
% end
% s21=s1/length;
% s22=s2/length;
% f=fullfile(fdir,'ParkScene*_HR.png');
% dirOutput=dir(f);
% filenames={dirOutput.name}';
% length = size(filenames,1);
% s1=0;
% s2=0;
% list1=[];
% list2=[];
% for i = 1: length
%     hr=imread(char(strcat(fdir, filenames(i))));
%     sr=imread(char(strcat(fdir,strrep(filenames(i),'_HR','_SR'))));
%     lr=imread(char(strcat(fdir,strrep(filenames(i),'_HR','_LR'))));
%     psnr_h_s=psnr(hr,sr);
%     psnr_h_l=psnr(hr,lr);
%     %fprintf('%d,%s,psnr_h_s=,%f,psnr_h_l=%f\n',i,char(filenames(i)),psnr_h_s,psnr_h_l);
%     s1 = s1+psnr_h_s;
%     s2 = s2+psnr_h_l;
%     list1=[list1,psnr_h_s];
%     list2=[list2,psnr_h_l];
% end
% s31=s1/length;
% s32=s2/length;
f=fullfile(fdir,'video13*_HR.png');
dirOutput=dir(f);
filenames={dirOutput.name}';
length = size(filenames,1);
s1=0;
s2=0;
list1=[];
list2=[];
for i = 1: length
    hr=imread(char(strcat(fdir, filenames(i))));
    sr=imread(char(strcat(fdir,strrep(filenames(i),'_HR','_SR'))));
    lr=imread(char(strcat(fdir,strrep(filenames(i),'_HR','_LR'))));
    psnr_h_s=psnr(hr,sr);
    psnr_h_l=psnr(hr,lr);
    %fprintf('%d,%s,psnr_h_s=,%f,psnr_h_l=%f\n',i,char(filenames(i)),psnr_h_s,psnr_h_l);
    s1 = s1+psnr_h_s;
    s2 = s2+psnr_h_l;
    list1=[list1,psnr_h_s];
    list2=[list2,psnr_h_l];
end
s41=s1/length
s42=s2/length
% fprintf('Bas:\npsnr_h_s=,%f,psnr_h_l=%f\nped:\npsnr_h_s=,%f,psnr_h_l=%f\nPar:\npsnr_h_s=,%f,psnr_h_l=%f\nKim:\npsnr_h_s=,%f,psnr_h_l=%f\n',s11,s12,s21,s22,s31,s32,s41,s42);