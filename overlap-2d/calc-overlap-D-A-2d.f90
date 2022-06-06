!   Bai Shuming 2019.12.26 Duke Univ: Updated to calcualte the overlap for D-B-A system, for which we need to calculate the Dp|Bp, Bp|Ap, Dp*|Bp*, Bp*|Ap*
!   Bai Shuming 2019.6.4 Duke University, Durham, NC, USA
!   Calculating the MO overlap from the cube file. The cube file has been transformed to (x,y,x,density) formet by cube2xyz program. 
!   Designed to calculate the overlap of HOMO-HOMO, LUMO-LUMO, and HOMO-HOMO*LUMO-LUMO. 

module params
implicit none
type den
 real*8  x,y,z
 integer snx, sny, snz
end type den
! type(den) trans(1000000)
end module params
 
program overlap_cube
use params
implicit none
integer i,j,k,nx,ny,nz,N,Nix,Npointx,Nhomo,Nlumo, natom, ncount, ncountdb, ncountba
integer N_br, nx_br, ny_br, nz_br, natom_br, Npointy, Niy
character*79:: record
real*8 stepx,stepy,stepz,trans_origin(3),dv1,transition(3),rotation(3,3)
real*8 stepx_br,stepy_br,stepz_br,trans_origin_br(3),dv1_br,transition_br(3),rotation_br(3,3)
real*8,allocatable :: overlapDhAh(:,:), overlapDlAl(:,:)
real*8,allocatable :: overlapDhBh(:,:), overlapDlBl(:,:), overlapBhAh(:,:), overlapBlAl(:,:)
real*8 x1,y1,z1,x2,y2,z2
integer nx1, ny1, nz1, nx2, ny2, nz2, n1,n2
real*8  coordx1, coordx2, coordy1, coordy2, coordz1, coordz2, theta, radius
real*8 density1,density2, couple, distance, couple0,couple1
real*8,parameter::au2cm=219474.63068d0
real*8,parameter::au2angstrom=0.529177249
real(8), parameter :: pi = 3.1415926535897932d0
real*8 tol_x,tol_y,tauol_zi, xyz_origin(3), xyz_origin_br(3), thresh
integer, allocatable :: if_takenl(:),if_takenh(:)
character*7 nonsense, ho_name, lu_name
character*7  ho_name_br, lu_name_br
real*8,allocatable :: homo_D(:)
real*8,allocatable :: homo_A(:)
real*8,allocatable :: lumo_D(:)
real*8,allocatable :: lumo_A(:)
real*8,allocatable :: homo_B(:)
real*8,allocatable :: lumo_B(:)


!input parameters---------------------------------------------------------------------------------------------
open(unit=111,file='input.dat')
read(111,*) ho_name, lu_name
read(111,*) ho_name_br, lu_name_br
read(111,*) Npointx, Npointy
read(111,*) trans_origin_br(1), trans_origin_br(2), trans_origin_br(3)
read(111,*) transition_br(1), transition_br(2), transition_br(3)
read(111,*) thresh
close(111)

trans_origin_br(:) = trans_origin_br(:)/au2angstrom
transition_br(:) = transition_br(:)/au2angstrom

write(*,*) "trans_origin_br", trans_origin_br(1),trans_origin_br(2), trans_origin_br(3)
write(*,*) "transition_br", transition_br(1),transition_br(2), transition_br(3)

    allocate(overlapDhAh(Npointx,Npointy))
    allocate(overlapDlAl(Npointx,Npointy))
    allocate(overlapDhBh(Npointx,Npointy))
    allocate(overlapBhAh(Npointx,Npointy))
    allocate(overlapDlBl(Npointx,Npointy))
    allocate(overlapBlAl(Npointx,Npointy))

! input MOs--------------------------------------------------------------------------------------------------
   open(unit=20,file=ho_name//".dat")
!       read(20,*) nonsense
       read(20,*) natom, xyz_origin(:)
       read(20,*) nx, stepx
       read(20,*) ny, stepy
       read(20,*) nz, stepz
       
       dv1 = stepx*stepy*stepz
       write(*,*) "dv1 = ", dv1

    N=nx*ny*nz

    allocate(homo_D(N))
    allocate(lumo_D(N))
    allocate(homo_A(N))
    allocate(lumo_A(N))

    do i = 1, N
       read(20,*) homo_D(i)
       homo_A(i) = homo_D(i)
    end do
    close(20)

    open(unit=30,file=lu_name//".dat")
!       read(30,*) nonsense
       read(30,*) natom, xyz_origin(:)
       read(30,*) nx, stepx
       read(30,*) ny, stepy
       read(30,*) nz, stepz  

    if(abs(stepx*stepy*stepz-dv1) .gt. 0.01) then
          write(*,*) "Wrong: MO1 and MO2 have different grid size!!!"
          stop
    end if
      
    do i = 1, N
       read(30,*) lumo_D(i)
       lumo_A(i) = lumo_D(i)
    end do 
    close(30)

! Input of the Bridge MOs-----------------------------------------------------
open(unit=20,file=ho_name_br//".dat")
!       read(20,*) nonsense
       read(20,*) natom_br, xyz_origin_br(:)
       read(20,*) nx_br, stepx_br
       read(20,*) ny_br, stepy_br
       read(20,*) nz_br, stepz_br

       dv1_br = stepx_br*stepy_br*stepz_br
       write(*,*) "dv1_bridge = ", dv1_br

    N_br=nx_br*ny_br*nz_br

    allocate(homo_B(N_br))
    allocate(lumo_B(N_br))

    do i = 1, N_br
       read(20,*) homo_B(i)
    end do
    close(20)

    open(unit=30,file=lu_name_br//".dat")
!       read(30,*) nonsense
       read(30,*) natom_br, xyz_origin_br(:)
       read(30,*) nx_br, stepx_br
       read(30,*) ny_br, stepy_br
       read(30,*) nz_br, stepz_br

    if(abs(stepx_br*stepy_br*stepz_br-dv1_br) .gt. 0.01) then
          write(*,*) "Wrong: MO1 and MO2 of Bridge have different grid size!!!"
          stop
    end if

    do i = 1, N_br
       read(30,*) lumo_B(i)
    end do
    close(3)
    write(*,*) "homo_b", "lumo_b", homo_B(10), lumo_B(10)

!Calculate the overlap=================================================================================================================== 
       overlapDhAh = 0.d0
       overlapDlAl = 0.d0
       overlapDhBh = 0.d0
       overlapBhAh = 0.d0
       overlapDlBl = 0.d0
       overlapBlAl = 0.d0

 open(555,file="overlap_"//ho_name//"_"//lu_name//"_"//ho_name_br//"_"//lu_name_br//".dat")
 write(555,*) "dX dY DhAh DlAl "

 do Nix =  1, Npointx
  do Niy = 1, Npointy
!      overlap between D and A:  
         ncount = 0
!      overlap between D and B:------------------------------------------------------------------------------------------------------------
         ncountdb = 0
         ncountba = 0
      do i = 1, nx_br
        do j = 1, ny_br
          do k = 1, nz_br
             n1 = (i-1)*ny_br*nz_br+(j-1)*nz_br+k
            if(abs(homo_B(n1)) .gt. thresh .or. abs(lumo_B(n1)) .gt. thresh) then
!              coordz1 = xyz_origin(3) + (k-1)*stepz
             x1 = xyz_origin_br(1)+(i-1)*stepx_br + trans_origin_br(1) + (Nix-1)*transition_br(1)
             nx1 = nint((x1-xyz_origin(1))/stepx)+1
             
             y1 = xyz_origin_br(2)+(j-1)*stepy_br + trans_origin_br(2) + (Niy-1)*transition_br(2)
             ny1 = nint((y1-xyz_origin(2))/stepy)+1

             z1 = xyz_origin_br(3)+(k-1)*stepz_br + trans_origin_br(3) + (Nix-1)*transition_br(3)
             nz1 = nint((z1-xyz_origin(3))/stepz)+1

!               if (k .eq. nz_br/2) then
!               write(*,*) i, j, k,nx1,ny1, nz1
!               write(*,*) nx2, ny2, nz2
!               end if 

              if (nx1 .ge. 1 .and. nx1 .le. nx) then
              if (ny1 .ge. 1 .and. ny1 .le. ny) then
              if (nz1 .ge. 1 .and. nz1 .le. nz) then
                  n2 = (nx1-1)*ny*nz+(ny1-1)*nz+nz1
                  overlapDhBh(Nix,Niy) = overlapDhBh(Nix,Niy) + homo_B(n1)*homo_D(n2)
                  overlapDlBl(Nix,Niy) = overlapDlBl(Nix,Niy) + lumo_B(n1)*lumo_D(n2)
!                  overlapl(Ni) = overlapl(Ni) + lumo(n1)*lumo(n2)
                  ncountdb = ncountdb + 1
              end if
              end if
              end if

             end if

           end do
         end do
       end do

!       write(*,*) "For cycle ", Ni, "number of D-B overlap points:", ncountdb
!        write(*,*) "For cycle ", Ni, "number of B-A overlap points:", ncountba

        write(555,*) (Nix-1)*transition_br(1)*au2angstrom, (Niy-1)*transition_br(2)*au2angstrom, &
                                   & overlapDhBh(Nix,Niy)*sqrt(dv1*dv1_br), &
                                     & overlapDlBl(Nix,Niy)*sqrt(dv1*dv1_br)
   end do
  end do


end 

