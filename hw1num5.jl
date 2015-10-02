
using Convex
using ECOS

n = 100
m = 30
A = randn(n,m)
b = randn(n,1)

x1 = Variable(m)
x2 = Variable(m)
x3 = Variable(m)
x4 = Variable(m)
println(x1)

p1 = minimize(norm(A*x1 - b, 1))
solve!(p1, ECOSSolver())

using PyPlot

figure(figsize = (20,5))
r = -2:0.1:2
plt[:hist](b-A*x1.value, 100, facecolor = "w")
plot(r, 13*abs(r), "k")
axis([-2,2,0,40]);

p2 = minimize(norm(A*x2 - b, 2))
solve!(p2, ECOSSolver())

using PyPlot

figure(figsize = (20,5))
r = -2:0.1:2
plt[:hist](b-A*x2.value, 100, facecolor = "w")
plot(r, 2*r.^2, "k")
axis([-2,2,0,10]);

p3 = minimize(sum(max(0,abs(A*x3 - b)-0.5)))
solve!(p3, ECOSSolver())

using PyPlot

figure(figsize = (20,5))
r = -2:0.1:2
plt[:hist](b-A*x3.value, 100, facecolor = "w")
plot(r, max(0,5*(abs(r)-0.5)), "k")
axis([-2,2,0,20]);

b2=0.5*b
p4 = minimize(sum(-log(1-A*x4+b2))-sum(log(1+A*x4-b2)), A*x4-b2<1, A*x4-b2>-1)
solve!(p4, ECOSSolver())

using PyPlot

figure(figsize = (20,5))
r = -0.99:0.01:0.99
u = -2:0.1:2
plt[:hist](b2-A*x4.value, 40, facecolor = "w")
plot(r, -3*(log(r+1) + log(-r + 1)), "k")
plot(u, 2*u.^2, "k--")
axis([-2,2,0,10]);


