% scatter(Date,High);
% hold on;
% scatter(Date,Peak,"red");
% scatter(Date,Valley,"blue");
% scatter(Date,Plain,'yellow');
% %scatter(Date,ForwardTotalElectricity);
% legend("High:8:30-11:00,16:30-21:00","Peak:10:00-11:00,19:00-21:00","Valley:12:00-13:00,23:00-7:00","Plain:21:00-23:00.11:00-12:00,13:00-14:30,21:00-23:00");
% ylabel("Daily Electricity Consumption (kWh)");
% xlabel("Date");
% hold off;

% histogram(High1);
% hold on;
% histogram(Peak1,'FaceColor',"r");
% histogram(Valley1,'FaceColor',"b");
% histogram(Plain1,'FaceColor','y');
% legend("High:8:30-11:00,16:30-21:00","Peak:10:00-11:00,19:00-21:00","Valley:12:00-13:00,23:00-7:00","Plain:21:00-23:00.11:00-12:00,13:00-14:30,21:00-23:00");
% xlabel("Daily Electricity Consumption (kWh)");
% ylabel("Count");
% title("September to November Electricity Use Histogram");
% hold off;

histogram(High2);
hold on;
histogram(Peak2,'FaceColor',"r");
histogram(Valley2,'FaceColor',"b");
histogram(Plain2,'FaceColor','y');
legend("High:8:30-11:00,16:30-21:00","Peak:10:00-11:00,19:00-21:00","Valley:12:00-13:00,23:00-7:00","Plain:21:00-23:00.11:00-12:00,13:00-14:30,21:00-23:00");
xlabel("Daily Electricity Consumption (kWh)");
ylabel("Count");
title("December to February Electricity Use Histogram");
hold off;


% histogram(High3);
% hold on;
% histogram(Peak3,'FaceColor',"r");
% histogram(Valley3,'FaceColor',"b");
% histogram(Plain3,'FaceColor','y');
% legend("High:8:30-11:00,16:30-21:00","Peak:10:00-11:00,19:00-21:00","Valley:12:00-13:00,23:00-7:00","Plain:21:00-23:00.11:00-12:00,13:00-14:30,21:00-23:00");
% xlabel("Daily Electricity Consumption (kWh)");
% ylabel("Count");
% title("March to May Electricity Use Histogram");
% hold off;

%summary(Data191011);
%summary(Data1345);
%summary(Data11212);

