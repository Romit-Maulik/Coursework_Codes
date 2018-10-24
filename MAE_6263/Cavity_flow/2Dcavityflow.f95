!This is an explicit solver for 1D Viscous Burgers Equation 2D NS for cavity flow
!CFD Lab OSU - Stillwater
!Romit Maulik,BS,MS,PhD (Underway)

program cavityflow2D

implicit none


integer::i,j,k,nx,ny,ns,nrec,smag,fostep
real*8::cvel,dx,dy,ft,rho,tl,dt,nu
real*8,allocatable,dimension(:,:)::u,v,p
real*8,allocatable,dimension(:,:,:)::urec,vrec,prec

common/ydivision/ny
common/xdivision/nx
common/xgrid/dx
common/ygrid/dy
common/cavityvel/cvel
common/viscosity/nu
common/density/rho
common/timestep/dt
common/smagmodel/ smag

open(7,file='2DNSInput.txt')
read(7,*)nx 	!number of spatial grid points x
read(7,*)ny		!number of spatial grip points y
read(7,*)ft		!total time step
read(7,*)rho	!Density
read(7,*)nu		!Viscosity
read(7,*)tl		!Total Length for square cavity
read(7,*)ns		!Time Steps in one iteration
read(7,*)cvel	!Lid velocity
read(7,*)smag	!Smagorinsky or Cebeci Modification
read(7,*)fostep	!File output after these many steps
close(7)


nrec = ns/100

allocate(u(0:nx,0:ny))
allocate(v(0:nx,0:ny))
allocate(p(0:nx,0:ny))
allocate(urec(0:nx,0:ny,0:nrec))
allocate(vrec(0:nx,0:ny,0:nrec))
allocate(prec(0:nx,0:ny,0:nrec))


dx = tl/dfloat(nx)
dy = tl/dfloat(ny)
dt = ft/dfloat(ns)

open(20,file="ContourPlots.plt")
write(20,*)'Title="Sample transient data set"'
write(20,*)'variables ="x","y","u","v","p"'
close(20)


call initcond(u,v,p)

do k = 1,ns

call udisc(u,v,p,i,j)

call vdisc(u,v,p,i,j)

do i = 0,nx
  do j = 0,ny    
if (u(i,j)*dt/dx + v(i,j)*dt/dy>1) then
  print*,k
  call exit(10)
end if
end do
end do

call pdisc(u,v,p,i,j)

if (mod(k,fostep)==0) then
open(20,file="ContourPlots.plt",position="append")
write(20,"(a,i8,a,i8,a)")'Zone I = ',nx+1,',J=',ny+1,',F=POINT'
write(20,"(a,i8)")'StrandID=0,SolutionTime=',k
  do j = 0,ny
    do i = 0,nx
      write (20, '(1600F14.3)',advance="no")dfloat(i),dfloat(j),u(i,j),v(i,j),p(i,j)
      write(20,*) ''
    end do
  end do
close(20)
end if

end do

end


!----------------------------------------------------------------
!Subroutine for initial conditions
!----------------------------------------------------------------
subroutine initcond(u,v,p)
implicit none

integer::nx,ny,i,j
real*8::cvel
common/ydivision/ny
common/xdivision/nx
common/cavityvel/cvel

real*8,dimension(0:nx,0:ny)::u,v,p

do j = 0,ny
  do i = 0,nx
    u(i,j) = 0
    v(i,j) = 0
    p(i,j) = 0
  end do
end do

do i = 0,nx
  u(i,ny) = cvel
end do


return
end

!----------------------------------------------------------------
!3rd Order RK for U momentum
!----------------------------------------------------------------

subroutine udisc(u,v,p,i,j)
implicit none


integer::nx,ny,j,i

common/ydivision/ny
common/xdivision/nx
common/cavityvel/cvel

real*8::cvel
real*8,dimension(0:nx,0:ny)::u,p,v,utemp,utemp1,utemp2,utemp3

do j = 0,ny-1
  do i = 0,nx
    utemp(i,j) = 0
    utemp1(i,j) = 0
    utemp2(i,j) = 0
    utemp3(i,j) = 0
  end do
end do

do i = 0,nx
    utemp(i,ny) = cvel
    utemp1(i,ny) = cvel
    utemp2(i,ny) = cvel
    utemp3(i,ny) = cvel
end do



do j = 1,ny-1
do i = 1,nx-1

call ulindisc(u,v,p,i,j,utemp)

utemp1(i,j) = u(i,j) + utemp(i,j)

end do
end do


do j = 1,ny-1
do i = 1,nx-1
  
call ulindisc(utemp1,v,p,i,j,utemp)

utemp2(i,j) = 0.75*u(i,j) + 0.25*utemp1(i,j) + 0.25*utemp(i,j)

end do
end do
  
  
do j = 1,ny-1
do i = 1,nx-1

call ulindisc(utemp2,v,p,i,j,utemp)

utemp3(i,j) = 1./3.*u(i,j) + 2./3.*utemp2(i,j) + 2./3.*utemp(i,j)

end do
end do



do j = 0,ny-1
  do i = 0,nx
	u(i,j) = utemp3(i,j)
  end do
end do

return
end

!----------------------------------------------------------------
!Linear Discretization Operator for u momentum equation
!----------------------------------------------------------------
subroutine ulindisc(a,b,c,i,j,temp)
implicit none

integer::nx,ny,i,j,smag
real*8::nu,rho,dt,dx,dy,smanu

common/ydivision/ny
common/xdivision/nx
common/viscosity/nu
common/density/rho
common/xgrid/dx
common/ygrid/dy
common/timestep/dt
common/smagmodel/ smag

real*8,dimension(0:nx,0:ny)::a,b,c,temp

smanu = 0.

if (smag==1) then
  call edviscsmag(a,b,i,j,smanu)
else if (smag==2) then
  call edviscebeci(a,b,i,j,smanu)
end if

temp(i,j) = -a(i,j)*dt/dx*(a(i,j)-a(i-1,j)) - b(i,j)*dt/dy*(a(i,j)-a(i,j-1))&
-dt/(rho*2.*dx)*(c(i+1,j)-c(i-1,j))+ (nu+smanu)*(dt/dx**2*(a(i+1,j)-2*a(i,j)+a(i-1,j))&
+dt/dy**2*(a(i,j+1)-2*a(i,j)+a(i,j-1)))


return
end

!----------------------------------------------------------------
!3rd Order RK for V momentum
!----------------------------------------------------------------

subroutine vdisc(u,v,p,i,j)
implicit none


integer::nx,ny,j,i
common/ydivision/ny
common/xdivision/nx

real*8,dimension(0:nx,0:ny)::u,p,v,vtemp,vtemp1,vtemp2,vtemp3

do j = 0,ny
  do i = 0,nx
    vtemp(i,j) = 0
    vtemp1(i,j) = 0
    vtemp2(i,j) = 0
    vtemp3(i,j) = 0
  end do
end do

do j = 1,ny-1
  do i = 1,nx-1
	call vlindisc(u,v,p,i,j,vtemp)
	vtemp1(i,j) = v(i,j) + vtemp(i,j)
  end do
end do


do j = 1,ny-1
  do i = 1,nx-1
	call vlindisc(u,vtemp1,p,i,j,vtemp)
	vtemp2(i,j) = 0.75*v(i,j) + 0.25*vtemp1(i,j) + 0.25*vtemp(i,j)
  end do
end do

do j = 1,ny-1
  do i = 1,nx-1
	call ulindisc(u,vtemp2,p,i,j,vtemp)
	vtemp3(i,j) = 1./3.*v(i,j) + 2./3.*vtemp2(i,j) + 2./3.*vtemp(i,j)
  end do
end do

do j = 0,ny
  do i = 0,nx
	v(i,j) = vtemp3(i,j)
  end do
end do


return
end

!----------------------------------------------------------------
!Linear Discretization Operator for v momentum equation
!----------------------------------------------------------------
subroutine vlindisc(a,b,c,i,j,temp)
implicit none

integer::nx,ny,i,j,smag
real*8::nu,rho,dt,dx,dy,smanu

common/ydivision/ny
common/xdivision/nx
common/viscosity/nu
common/density/rho
common/xgrid/dx
common/ygrid/dy
common/timestep/dt
common/smagmodel/ smag

real*8,dimension(0:nx,0:ny)::a,b,c,temp

smanu = 0.

if (smag==1) then
  call edviscsmag(a,b,i,j,smanu)
else if (smag==2) then
  call edviscebeci(a,b,i,j,smanu)  
end if

temp(i,j) = -a(i,j)*dt/dx*(b(i,j)-b(i-1,j)) - b(i,j)*dt/dy*(b(i,j)-b(i,j-1))&
-dt/(rho*2.*dy)*(c(i,j+1)-c(i,j-1))+ (nu+smanu)*(dt/dx**2*(b(i+1,j)-2*b(i,j)+b(i-1,j))&
+dt/dy**2*(b(i,j+1)-2*b(i,j)+b(i,j-1)))



return
end


!----------------------------------------------------------------
!Discretization Operator for Pressure Equation
!----------------------------------------------------------------

subroutine pdisc(a,b,c,i,j)
implicit none

integer::nx,ny,i,j
real*8::dx,dy,rho,dt,parta,partb,partc,partd,parte,partf

common/ydivision/ny
common/xdivision/nx
common/xgrid/dx
common/ygrid/dy
common/density/rho
common/timestep/dt

real*8,dimension(0:nx,0:ny)::a,b,c

do j = 0,ny
	do i = 0,nx

if (i==0) then
  c(i,j) = c(i+1,j)
else if (i==nx) then
  c(nx,j) = c(nx-1,j)
else if (j==ny) then
  c(i,j) = 0
else if (j==0) then
  c(i,j) = c(i,j+1)
else
  
parta = ((c(i+1,j)+c(i-1,j))*dy**2+(c(i,j+1)+c(i,j-1))*dx**2)/(2*(dx**2+dy**2))
partb = (rho*dx**2*dy**2)/(2*(dx**2+dy**2))
partc = (1./dt)*((a(i+1,j) - a(i-1,j))/(2*dx)+(b(i,j+1) - b(i,j-1))/(2*dy))
partd = (a(i+1,j)-a(i-1,j))/(2*dx)*(a(i+1,j)-a(i-1,j))/(2*dx)
parte = 2*(a(i,j+1)-a(i,j-1))/(2*dy)*(b(i+1,j)-b(i-1,j))/(2*dx)
partf = (b(i,j+1)-b(i,j-1))/(2*dy)*(b(i,j+1)-b(i,j-1))/(2*dy)

c(i,j) = parta - partb*(partc - partd - parte - partf)

end if
  
	end do
end do




return
end



!---------------------------------------------------------------------
! Smagorinsky Method to calculate eddy viscosity
!---------------------------------------------------------------------
subroutine edviscsmag(u,v,i,j,smanu)
implicit none

integer::nx,ny,i,j
real*8::smanu,dx,dy,smagcoeff,delg,gradsq,t1,t2,t3,t4


common/ydivision/ny
common/xdivision/nx
common/xgrid/dx
common/ygrid/dy

real*8,dimension(0:nx,0:ny)::u,v

t1 = ((u(i+1,j)-u(i-1,j))/(2.*dx))**2
t4 = ((v(i,j+1)-v(i,j-1))/(2.*dy))**2
t2 = (0.5d0*((u(i,j+1)-u(i,j-1))/(2.*dy)+(v(i+1,j)-v(i-1,j))/(2.*dx)))**2
t3 = (0.5d0*((v(i+1,j)-v(i-1,j))/(2.*dx)+(u(i,j+1)-u(i,j-1))/(2.*dy)))**2

smagcoeff = 0.1782d0
delg = (dx*dy)**(1.0d0/3.0d0)



gradsq = dsqrt(2.*(t1+t2+t3+t4))
smanu = ((smagcoeff**2*delg)**2)*gradsq


return
end

!---------------------------------------------------------------------
! Cebeci Smith Method to calculate eddy viscosity
!---------------------------------------------------------------------

subroutine edviscebeci(u,v,i,j,smanu)
implicit none

integer::nx,ny,i,j
real*8::smanu,dx,dy,t1,t2,t3,t4,smagcoeff,h


common/ydivision/ny
common/xdivision/nx
common/xgrid/dx
common/ygrid/dy

real*8,dimension(0:nx,0:ny)::u,v

!ux
t1 = ((u(i+1,j)-u(i-1,j))/(2.*dx))**2

!uy
t2 = ((u(i,j+1)-u(i,j-1))/(2.*dy))**2

!vx
t3 = ((v(i+1,j)-v(i-1,j))/(2.*dx))**2

!vy
t4 = ((v(i,j+1)-v(i,j+1))/(2.*dy))**2

smanu = t1+t2+t3+t4

smagcoeff = 0.2d0

h = (dx*dy)**(1.0d0/3.0d0)

smanu = (smagcoeff**2)*(h**2)*dsqrt(smanu)

return
end
