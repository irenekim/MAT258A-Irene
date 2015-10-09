
Admissions = readdlm("/Users/iRene/Desktop/2015fall/MAT258A/hw2/binary.csv", ',', header=true);

y = Admissions[1][1:400,1];

u = Admissions[1][1:400,2:3]*float(Diagonal([1/200, 1]));

v = [1 for i = 1:400];

nu = cat(2,u,v);

nv = 0
for i in 1:400
    v = (nu[i,1:3]')*(nu[i,1:3])
    nv = nv + v
end

eigvals(nv)

L = 8568.38 /4

function f(x) 
    sum([-y[i]*dot(vec(nu[i,1:3]),x)+log(1+exp(dot(vec(nu[i,1:3]),x))) for i = 1:400])
end

function gradf(x)
    M = zeros(3,400)
    for i = 1:400
        M[:,i] = [-y[i]*vec(nu[i,1:3]')+vec(nu[i,1:3]')*exp(dot(vec(nu[i,1:3]),x))/(1+exp(dot(vec(nu[i,1:3]),x)))]
    end
    vec(sum(M,2))
end   

x = zeros(3)
fk = f(x)
for itn = 1:20000
    x = x - (1/L)*gradf(x)
    fk = [fk;f(x)]
end
[x, fk]

x = [ 0,0.8,-5 ] # line search algorithm
c = 1/2
alpha = 1
fk = f(x)
for itn = 1:1000
    while f(x) - f(x - alpha*gradf(x))<alpha*c*(norm(gradf(x)))^2 
        alpha = (1/10)*alpha
    end
    x = x - alpha*gradf(x)
    fk = [fk;f(x)]
end
[x, fk]

myarr = Admissions[1];

admit = myarr[myarr[:,1].==1, 2:3];
reject = myarr[myarr[:,1].==0, 2:3];

using PyPlot

plot(admit[:,1], admit[:,2], "+")
plot(reject[:,1], reject[:,2], "o",
markerfacecolor = "None"
)
xlabel("GRE")
ylabel("GPA")
title("GRE vs GPA")
axis([100,900, 2,4.2])

plot(admit[:,1], admit[:,2], "+")
plot(reject[:,1], reject[:,2], "o",
markerfacecolor = "None"
)
axis([100,1100, 2,7])
x = [100:0.01:1100] #GRE
y = (-0.00252069*x+5.00002+0.5)/0.799914 #GPA
plot(x,y)
xlabel("GRE")
ylabel("GPA")
title("Decision Boundary")


