!Course Project II for MAE 6263
!Explicit and Implicit Solver for 2-D Burgers' Equation
!Implicit Solver uses Jacobi Iterative Method
!ICs - u(x,y) = 0 everywhere except boundaries
!BCs - u(x,1) = 1 through time
!Romit Maulik, PhD Student, CFD Laboratory, OSU

program CFDCP2
implicit none


integer :: nx,ny,gsnt,isol,fostep,i,j,ns,k,outpar,adsc,itsol
real*8 :: ft,xl,yl,nu,tol,dt,dx,dy,t1,t2
real*8 :: cflu,cflv,regl,pedx,lidv,gm
real*8,allocatable :: u(:,:),v(:,:)

common/viscosity/ nu
common/adscheme/ adsc
common/impiter/ gsnt,itsol
common/contol/ tol

open(15,file='CFDCP2Input.txt')
read(15,*)pedx		!Cell Peclet number
read(15,*)regl		!Global Reynolds number
read(15,*)lidv		!Lid Velocity
read(15,*)gm		!Viscous courant
read(15,*)xl		!Total Length in x direction
read(15,*)yl		!Total Length in y direction
read(15,*)ft		!Final Time
read(15,*)isol		!isol;	Solver - [0] - For explicit [1] - For Implicit [2] - Semi Implicit
read(15,*)itsol		!itsol; Iterative solver - [0] - Gauss-Jacobi [1] - Gauss Siedel
read(15,*)gsnt		!gsnt; 	Number of Iterative Steps in GS or Jacobi
read(15,*)adsc		!adsc; Advection scheme [0] for Upwind [1] - Central Second Order
read(15,*)fostep	!fostep;File output every this percent of total timesteps(Choose multiples of ten)
read(15,*)tol		!tol; Convergence tolerance
close(15)


!Calculating viscosity
nu = lidv*yl/regl

!Calculating dx and dy
dx = pedx*nu/lidv
dy = dx

!Calculating timestep
dt = gm*(dx**2)/nu

!Calculating number of timesteps
ns = nint(ft/dt)

!Calculating number of grid discretizations
nx = nint(xl/dx)
ny = nint(yl/dy)

!Initial Condition & Boundary Condition Setup
allocate(u(0:nx,0:ny))
allocate(v(0:nx,0:ny))

do i = 0,nx
  do j = 0,ny-1
    u(i,j) = 0
    v(i,j) = 0
  end do
	
  u(i,ny) = lidv
  v(i,ny) = 0
end do


!IC File Ouput
open(20,file="InitialField.plt")
write(20,*)'Title="IC Data set"'
write(20,*)'variables ="x","y","u","v"'
close(20)

open(20,file="InitialField.plt",position="append")
write(20,"(a,i8,a,i8,a)")'Zone I = ',nx+1,',J=',ny+1,',F=POINT'
  do j = 0,ny
    do i = 0,nx
      write (20, '(1600F14.3)',advance="no")dfloat(i)/dfloat(ny),dfloat(j)/dfloat(ny),u(i,j),v(i,j)
      write(20,*) ''
    end do
  end do
close(20)



!Output file setup
open(20,file="ContourPlots.plt")
write(20,*)'Title="Transient data set"'
write(20,*)'variables ="x","y","u","v"'
close(20)

!Output file setup at y=0.5
open(20,file="LinePlots.plt")
write(20,*)'Title="Transient data set"'
write(20,*)'variables ="y","u"'
close(20)

call cpu_time(t1)
!Time integration - 
do k = 1,ns

  
!Stability Check
if (isol == 0) then
	cflu = maxval(u)
    cflv = maxval(v) 
if ((cflu*dt/(dx)+cflv*dt/(dy))>1d0) then
  print*,'Unstable - Reduce Timestep'
  print*,dt
!  call exit(10)
end if
end if

if (isol == 0) then
	call bgfullexp(u,v,nx,ny,dx,dy,dt)	
else if (isol == 1) then
  	call bgfullimp(u,v,nx,ny,dx,dy,dt)
else if (isol == 2) then
  	call bgexpimp(u,v,nx,ny,dx,dy,dt)
end if



outpar = ns*fostep/100

  
!Output to transient .plt file for Tecplot  
!if (mod(k,outpar)==0) then
open(20,file="ContourPlots.plt",position="append")
write(20,"(a,i8,a,i8,a)")'Zone I = ',nx+1,',J=',ny+1,',F=POINT'
write(20,"(a,i8)")'StrandID=0,SolutionTime=',k
  do j = 0,ny
    do i = 0,nx
      write (20, '(1600F14.3,1600F14.3,1600F14.3,1600F14.3)',advance="no")dfloat(i)/dfloat(ny),dfloat(j)/dfloat(ny),u(i,j),v(i,j)
      write(20,*) ''
    end do
  end do
close(20)

open(20,file="LinePlots.plt",position="append")
write(20,"(a,i8,a)")'Zone I = ',ny+1,',F=POINT'
write(20,"(a,i8)")'StrandID=0,SolutionTime=',k
    do j = 0,ny
      write (20, '(1600F14.3,1600F14.3)',advance="no")dfloat(j)/dfloat(ny),u(nx/2,j)
      write(20,*) ''
    end do
close(20)

!end if


 
end do

call cpu_time(t2)

open(4,file='cpu.txt')
write(4,*)"cpu time (sec)=",(t2-t1)
close(4)


end


!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
!Subroutine for Explicit Solver - Verified
!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
subroutine bgfullexp(u,v,nx,ny,dx,dy,dt)
implicit none

integer :: nx,ny,i,j,adsc
real*8 :: dt,dx,dy,nu,t1,t2,t3,t4
real*8,dimension(0:nx,0:ny):: u,v
real*8,dimension(-1:nx+1,0:ny) :: utemp,vtemp,usol,vsol

common/viscosity/ nu
common/adscheme/ adsc

!Explicit algebraic solving - point to point

do i = 0,nx
  do j = 0,ny
	utemp(i,j) = u(i,j)
    vtemp(i,j) = v(i,j)
    usol(i,j) = u(i,j)
    vsol(i,j) = v(i,j)
  end do
end do

do j = 0,ny
utemp(-1,j) = u(nx-1,j)
vtemp(-1,j) = v(nx-1,j)
utemp(nx+1,j) = u(1,j)
vtemp(nx+1,j) = v(1,j)
usol(-1,j) = u(nx-1,j)
vsol(-1,j) = v(nx-1,j)
usol(nx+1,j) = u(1,j)
vsol(nx+1,j) = v(1,j)
end do

do j = 1,ny-1
  do i = 0,nx
          
    if (adsc==0) then
	    t1 = usol(i,j)*(usol(i,j)-usol(i-1,j))/dx
	    t2 = vsol(i,j)*(usol(i,j)-usol(i,j-1))/dy        
    else if (adsc==1) then
		t1 = usol(i,j)*(usol(i+1,j)-usol(i-1,j))/(2d0*dx)
	    t2 = vsol(i,j)*(usol(i,j+1)-usol(i,j-1))/(2d0*dy)        
    end if  

    t3 = nu*(usol(i+1,j)+usol(i-1,j)-2.*usol(i,j))/(dx**2)
    t4 = nu*(usol(i,j+1)+usol(i,j-1)-2.*usol(i,j))/(dy**2)
    utemp(i,j) = usol(i,j) + dt*(t3+t4-t1-t2)
  end do
end do

do j = 1,ny-1
  do i = 0,nx

    if (adsc==0) then
	    t1 = usol(i,j)*(vsol(i,j)-vsol(i-1,j))/dx
    	t2 = vsol(i,j)*(vsol(i,j)-vsol(i,j-1))/dy
    else if (adsc==1) then
	    t1 = usol(i,j)*(vsol(i+1,j)-vsol(i-1,j))/(2d0*dx)
    	t2 = vsol(i,j)*(vsol(i,j+1)-vsol(i,j-1))/(2d0*dy)
    end if  


    t3 = nu*(vsol(i,j+1)+vsol(i,j-1)-2.*vsol(i,j))/(dy**2)
    t4 = nu*(vsol(i+1,j)+vsol(i-1,j)-2.*vsol(i,j))/(dx**2)    
    vtemp(i,j) = vsol(i,j) + dt*(t3+t4-t1-t2)
  end do
end do



do j = 1,ny-1
  do i = 0,nx
    u(i,j) = utemp(i,j)
    v(i,j) = vtemp(i,j)
  end do
end do

return
end

!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
!Subroutine for Fully Implicit Solver - Jacobi Iterations - Verified
!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
subroutine bgfullimp(u,v,nx,ny,dx,dy,dt)
implicit none

integer :: nx,ny,i,j,adsc,k,q,gsnt,itsol
real*8 :: dt,dx,dy,nu,t1,t2,t3,t4
real*8,dimension(0:nx,0:ny):: u,v
real*8,dimension(-1:nx+1,0:ny):: usol,vsol
real*8,dimension(-1:nx+1,0:ny) :: utemp,vtemp

common/viscosity/ nu
common/adscheme/ adsc
common/impiter/ gsnt,itsol


do i = 0,nx
  do j = 0,ny
	utemp(i,j) = u(i,j)
    vtemp(i,j) = v(i,j)
	usol(i,j) = u(i,j)
    vsol(i,j) = v(i,j)    
  end do
end do

if (itsol==0) then  !Gauss Jacobi Method

do k = 1,gsnt

do j = 0,ny
usol(-1,j) = usol(nx-1,j) 
utemp(-1,j) = usol(nx-1,j)
vsol(-1,j) = vsol(nx-1,j)   
vtemp(-1,j) = vsol(nx-1,j)
usol(nx+1,j) = usol(1,j)
utemp(nx+1,j) = usol(1,j)
vsol(nx+1,j) = vsol(1,j)
vtemp(nx+1,j) = vsol(1,j)
end do  

do j = 1,ny-1
  do i = 0,nx

    	if (adsc==0) then
    		t1 = 1+(dt*usol(i,j)/dx) - (dt*usol(i-1,j)/dx) + (dt*vsol(i,j)/dy) + 2.*dt*nu/(dx**2)+2.*dt*nu/(dy**2)
    		t2 = dt*vsol(i,j)*usol(i,j-1)/dy
        else if (adsc==1) then
        	t1 = 1 + 2.*dt*nu/(dx**2)+ 2.*dt*nu/(dy**2) + dt/(2d0*dx)*(usol(i+1,j)-usol(i-1,j))
    		t2 = -dt*vsol(i,j)/(2d0*dy)*(usol(i,j+1)-usol(i,j-1))
		end if
        
    	t3 = dt*nu/(dx**2)*(usol(i+1,j)+usol(i-1,j))
    	t4 = dt*nu/(dy**2)*(usol(i,j+1)+usol(i,j-1))    
    	utemp(i,j) = (u(i,j) + t2 + t3 + t4)/(t1)

  end do
end do


do j = 1,ny-1
  do i = 0,nx
    if (adsc==0) then
      	t1 = 1+(dt*usol(i,j)/dx) + vsol(i,j)*dt/dy - vsol(i,j-1)*dt/dy  + 2.*dt*nu/(dx**2)+2.*dt*nu/(dy**2)
    	t2 = dt*usol(i,j)*vsol(i-1,j)/dx
    else if (adsc==1) then
      	t1 = 1+ 2.*dt*nu/(dx**2) + 2.*dt*nu/(dy**2) + dt/(2d0*dy)*(vsol(i,j+1)-vsol(i,j-1))
    	t2 = -dt/(2d0*dx)*(vsol(i+1,j)-vsol(i-1,j))
    end if  
    t3 = dt*nu/(dx**2)*(vsol(i+1,j)+vsol(i-1,j))
    t4 = dt*nu/(dy**2)*(vsol(i,j+1)+vsol(i,j-1))    
    vtemp(i,j) = (v(i,j) + t2 + t3 + t4)/(t1)
  end do
end do

q = 0
call l1normcheck(utemp,vtemp,usol,vsol,nx,ny,q)

if (q.ne.0) then
  	do j = 1,ny-1
  	do i = 0,nx
    	u(i,j) = utemp(i,j)
    	v(i,j) = vtemp(i,j)
  	end do
	end do
    print*,k
  exit

else if (k==gsnt) then
  	do j = 1,ny-1
  	do i = 0,nx
    	u(i,j) = utemp(i,j)
    	v(i,j) = vtemp(i,j)
  	end do
	end do
    print*,k    

 end if
  
do j = 1,ny-1
  do i = 0,nx
    usol(i,j) = utemp(i,j)
    vsol(i,j) = vtemp(i,j)
  end do
end do

end do

else if (itsol==1) then !Gauss Siedel Iterative Method

do k = 1,gsnt


do j = 0,ny
usol(-1,j) = usol(nx-1,j) 
utemp(-1,j) = utemp(nx-1,j)
vsol(-1,j) = vsol(nx-1,j)   
vtemp(-1,j) = vtemp(nx-1,j)
usol(nx+1,j) = usol(1,j)
utemp(nx+1,j) = utemp(1,j)
vsol(nx+1,j) = vsol(1,j)
vtemp(nx+1,j) = vtemp(1,j)
end do
  
do j = 1,ny-1
  do i = 0,nx

   	if (adsc==0) then
    		t1 = 1+ 2.*dt*nu/(dx**2)+ 2.*dt*nu/(dy**2)+ (dt*usol(i,j)/dx) - (dt*utemp(i-1,j)/dx) + (dt*vsol(i,j)/dy) 
    		t2 = dt*vsol(i,j)*utemp(i,j-1)/dy
        else if (adsc==1) then
        	t1 = 1 + 2.*dt*nu/(dx**2)+ 2.*dt*nu/(dy**2) + dt/(2d0*dx)*(usol(i+1,j)-utemp(i-1,j))
    		t2 = -dt*vsol(i,j)/(2d0*dy)*(usol(i,j+1)-utemp(i,j-1))
		end if
       
    	t3 = dt*nu/(dx**2)*(usol(i+1,j)+utemp(i-1,j))
    	t4 = dt*nu/(dy**2)*(usol(i,j+1)+utemp(i,j-1))    
    	utemp(i,j) = (u(i,j) + t2 + t3 + t4)/(t1)


  end do

  utemp(-1,j) = utemp(nx-1,j)
  utemp(nx+1,j) = utemp(1,j)
  
end do


do j = 1,ny-1
  do i = 0,nx
    if (adsc==0) then
      	t1 = 1+(dt*usol(i,j)/dx) + vsol(i,j)*dt/dy - vtemp(i,j-1)*dt/dy  + 2.*dt*nu/(dx**2)+2.*dt*nu/(dy**2)
    	t2 = dt*usol(i,j)*vtemp(i-1,j)/dx
    else if (adsc==1) then
      	t1 = 1+ 2.*dt*nu/(dx**2) + 2.*dt*nu/(dy**2) + dt/(2d0*dy)*(vsol(i,j+1)-vtemp(i,j-1))
    	t2 = -dt/(2d0*dx)*(vsol(i+1,j)-vtemp(i-1,j))
    end if  
    t3 = dt*nu/(dx**2)*(vsol(i+1,j)+vtemp(i-1,j))
    t4 = dt*nu/(dy**2)*(vsol(i,j+1)+vtemp(i,j-1))    
    vtemp(i,j) = (v(i,j) + t2 + t3 + t4)/(t1)
  end do

  
  vtemp(-1,j) = vtemp(nx-1,j)
  vtemp(nx+1,j) = vtemp(1,j)
end do

q = 0
call l1normcheck(utemp,vtemp,usol,vsol,nx,ny,q)

do i = 0,nx
  do j = 0,ny
	usol(i,j) = utemp(i,j)
    vsol(i,j) = vtemp(i,j)
  end do
end do 

  
if (q.ne.0) then
  	do j = 1,ny-1
  	do i = 0,nx
    	u(i,j) = usol(i,j)
    	v(i,j) = vsol(i,j)
    end do
	end do
    print*,k    
  exit

else if (k==gsnt) then
  	do j = 1,ny-1
  	do i = 0,nx
    	u(i,j) = usol(i,j)
    	v(i,j) = vsol(i,j)
  	end do
	end do
    print*,k    
  
end if

end do  


end if


return 
end





!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
!Subroutine for part explicit and part implicit Solver - Jacobi Iterations
!The Viscous dissipation is implicit and the advection is explicit
!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
subroutine bgexpimp(u,v,nx,ny,dx,dy,dt)
implicit none

integer :: nx,ny,i,j,adsc,gsnt,q,k,itsol
real*8 :: dt,dx,dy,nu,t1,t2,t3,t4
real*8,dimension(0:nx,0:ny):: u,v
real*8,dimension(-1:nx+1,0:ny) :: utemp,vtemp,usol,vsol,usol1,vsol1

common/viscosity/ nu
common/adscheme/ adsc
common/impiter/ gsnt,itsol


if (itsol==0) then   !Gauss-Jacobi method

do i = 0,nx
  do j = 0,ny
	utemp(i,j) = u(i,j)
    vtemp(i,j) = v(i,j)
	usol(i,j) = u(i,j)
    vsol(i,j) = v(i,j)
	usol1(i,j) = u(i,j)
    vsol1(i,j) = v(i,j)        
  end do
end do
  
do k = 1,gsnt


do j = 0,ny
usol(-1,j) = usol(nx-1,j)
usol1(-1,j) = usol1(nx-1,j) 
utemp(-1,j) = usol(nx-1,j)
vsol(-1,j) = vsol(nx-1,j)
vsol1(-1,j) = vsol1(nx-1,j)   
vtemp(-1,j) = vsol(nx-1,j)
usol(nx+1,j) = usol(1,j)
usol1(nx+1,j) = usol1(1,j)
utemp(nx+1,j) = usol(1,j)
vsol(nx+1,j) = vsol(1,j)
vsol1(nx+1,j) = vsol1(1,j)
vtemp(nx+1,j) = vsol(1,j)
end do

do j = 1,ny-1
  do i = 0,nx

    if (adsc==0) then
	    t1 = 1 + 2.*dt*nu/(dx**2)+2.*dt*nu/(dy**2)
    	t2 = -dt*usol1(i,j)*(usol1(i,j)-usol1(i-1,j))/dx - dt*vsol1(i,j)*(usol1(i,j)-usol1(i,j-1))/dy
    else if (adsc==1) then
    	t1 = 1+ 2.*dt*nu/(dx**2)+2.*dt*nu/(dy**2)
    	t2 = -dt*usol1(i,j)*(usol1(i+1,j)-usol1(i-1,j))/2d0*dx - dt*vsol1(i,j)*(usol1(i,j+1)-usol1(i,j-1))/2d0*dy
    end if

    t3 = dt*nu/(dx**2)*(usol(i+1,j)+usol(i-1,j))
    t4 = dt*nu/(dy**2)*(usol(i,j+1)+usol(i,j-1))    
    utemp(i,j) = (u(i,j) + t2 + t3 + t4)/(t1)
  end do
end do


do j = 1,ny-1
  do i = 0,nx

    if (adsc==0) then
	    t1 = 1+ 2.*dt*nu/(dx**2)+2.*dt*nu/(dy**2)
    	t2 = -dt*usol1(i,j)*(vsol1(i,j)-vsol1(i-1,j))/dx - dt*vsol1(i,j)*(vsol1(i,j)-vsol1(i,j-1))/dy
    else if (adsc==1) then
	    t1 = 1+ 2.*dt*nu/(dx**2)+2.*dt*nu/(dy**2)
    	t2 = -dt*usol1(i,j)*(vsol1(i+1,j)-vsol1(i-1,j))/dx - dt*vsol1(i,j)*(vsol1(i,j+1)-vsol1(i,j-1))/dy
    end if  


    t3 = dt*nu/(dx**2)*(vsol(i+1,j)+vsol(i-1,j))
    t4 = dt*nu/(dy**2)*(vsol(i,j+1)+vsol(i,j-1))    
    vtemp(i,j) = (v(i,j) + t2 + t3 + t4)/(t1)
  end do
end do


q = 0
call l1normcheck(utemp,vtemp,usol,vsol,nx,ny,q)

do j = 1,ny-1
  do i = 0,nx
    usol(i,j) = utemp(i,j)
    vsol(i,j) = vtemp(i,j)
  end do
end do

if (q.ne.0) then
    do j = 1,ny-1
  	do i = 0,nx
    	u(i,j) = utemp(i,j)
    	v(i,j) = vtemp(i,j)
    end do
	end do
    print*,k
  exit

else if (k==gsnt) then
  	do j = 1,ny-1
  	do i = 0,nx
    	u(i,j) = utemp(i,j)
    	v(i,j) = vtemp(i,j)
  	end do
	end do
    print*,k    
 
end if  


end do

else if (itsol==1) then !Gauss-Siedel method

do i = 0,nx
  do j = 0,ny
	utemp(i,j) = u(i,j)
    vtemp(i,j) = v(i,j)
	usol(i,j) = u(i,j)
    vsol(i,j) = v(i,j)
	usol1(i,j) = u(i,j)
    vsol1(i,j) = v(i,j)        
  end do
end do
  
do k = 1,gsnt


do j = 0,ny
usol(-1,j) = usol(nx-1,j)
usol1(-1,j) = usol1(nx-1,j) 
utemp(-1,j) = usol(nx-1,j)
vsol(-1,j) = vsol(nx-1,j)
vsol1(-1,j) = vsol1(nx-1,j)   
vtemp(-1,j) = vsol(nx-1,j)
usol(nx+1,j) = usol(1,j)
usol1(nx+1,j) = usol1(1,j)
utemp(nx+1,j) = usol(1,j)
vsol(nx+1,j) = vsol(1,j)
vsol1(nx+1,j) = vsol1(1,j)
vtemp(nx+1,j) = vsol(1,j)
end do

do j = 1,ny-1
  do i = 0,nx

    if (adsc==0) then
	    t1 = 1+ 2.*dt*nu/(dx**2)+2.*dt*nu/(dy**2)
    	t2 = -dt*usol1(i,j)*(usol1(i,j)-usol1(i-1,j))/dx - dt*vsol1(i,j)*(usol1(i,j)-usol1(i,j-1))/dy
    else if (adsc==1) then
    	t1 = 1+ 2.*dt*nu/(dx**2)+2.*dt*nu/(dy**2)
    	t2 = -dt*usol1(i,j)*(usol1(i+1,j)-usol1(i-1,j))/2d0*dx - dt*v(i,j)*(usol1(i,j+1)-usol1(i,j-1))/2d0*dy
    end if

    t3 = dt*nu/(dx**2)*(usol(i+1,j)+utemp(i-1,j))
    t4 = dt*nu/(dy**2)*(usol(i,j+1)+utemp(i,j-1))    
    utemp(i,j) = (u(i,j) + t2 + t3 + t4)/(t1)
  end do
end do


do j = 1,ny-1
  do i = 0,nx

    if (adsc==0) then
	    t1 = 1+ 2.*dt*nu/(dx**2)+2.*dt*nu/(dy**2)
    	t2 = -dt*usol1(i,j)*(vsol1(i,j)-vsol1(i-1,j))/dx - dt*vsol1(i,j)*(vsol1(i,j)-vsol1(i,j-1))/dy
    else if (adsc==1) then
	    t1 = 1+ 2.*dt*nu/(dx**2)+2.*dt*nu/(dy**2)
    	t2 = -dt*usol1(i,j)*(vsol1(i+1,j)-vsol1(i-1,j))/dx - dt*vsol1(i,j)*(vsol1(i,j+1)-vsol1(i,j-1))/dy
    end if  


    t3 = dt*nu/(dx**2)*(vsol(i+1,j)+vtemp(i-1,j))
    t4 = dt*nu/(dy**2)*(vsol(i,j+1)+vtemp(i,j-1))    
    vtemp(i,j) = (v(i,j) + t2 + t3 + t4)/(t1)
  end do
end do


q = 0
call l1normcheck(utemp,vtemp,usol,vsol,nx,ny,q)

do j = 1,ny-1
  do i = 0,nx
    usol(i,j) = utemp(i,j)
    vsol(i,j) = vtemp(i,j)
  end do
end do

if (q.ne.0) then
    do j = 1,ny-1
  	do i = 0,nx
    	u(i,j) = utemp(i,j)
    	v(i,j) = vtemp(i,j)
    end do
	end do
    print*,k    
  exit

else if (k==gsnt) then
  	do j = 1,ny-1
  	do i = 0,nx
    	u(i,j) = utemp(i,j)
    	v(i,j) = vtemp(i,j)
  	end do
	end do
    print*,k    
  
end if


end do





end if  
  

return
end





!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
!Subroutine for L1 Norm Check
!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
subroutine l1normcheck(utemp,vtemp,u,v,nx,ny,q)
implicit none

common/contol/ tol

integer::q,nx,ny,i,j
real*8,dimension(-1:nx+1,0:ny)::utemp,vtemp,u,v
real*8::sumu,sumv,tol

sumu = 0.

do j = 0,ny
	do i = 0,nx
  		sumu = sumu + (utemp(i,j)-u(i,j))
	end do
end do

sumv = 0.

do j = 0,ny
	do i = 0,nx
  		sumv = sumv + (vtemp(i,j)-v(i,j))
	end do
end do

if (sumu<tol.and.sumv<tol) then
  q = 1
end if

return
end