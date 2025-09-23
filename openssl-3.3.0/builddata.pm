package OpenSSL::safe::installdata;

use strict;
use warnings;
use Exporter;
our @ISA = qw(Exporter);
our @EXPORT = qw($PREFIX
                  $BINDIR $BINDIR_REL
                  $LIBDIR $LIBDIR_REL
                  $INCLUDEDIR $INCLUDEDIR_REL
                  $APPLINKDIR $APPLINKDIR_REL
                  $ENGINESDIR $ENGINESDIR_REL
                  $MODULESDIR $MODULESDIR_REL
                  $PKGCONFIGDIR $PKGCONFIGDIR_REL
                  $CMAKECONFIGDIR $CMAKECONFIGDIR_REL
                  $VERSION @LDLIBS);

our $PREFIX             = '/mnt/c/Users/user1/Desktop/src/openssl-3.3.0';
our $BINDIR             = '/mnt/c/Users/user1/Desktop/src/openssl-3.3.0/apps';
our $BINDIR_REL         = 'apps';
our $LIBDIR             = '/mnt/c/Users/user1/Desktop/src/openssl-3.3.0';
our $LIBDIR_REL         = '.';
our $INCLUDEDIR         = '/mnt/c/Users/user1/Desktop/src/openssl-3.3.0/include';
our $INCLUDEDIR_REL     = 'include';
our $APPLINKDIR         = '/mnt/c/Users/user1/Desktop/src/openssl-3.3.0/ms';
our $APPLINKDIR_REL     = 'ms';
our $ENGINESDIR         = '/mnt/c/Users/user1/Desktop/src/openssl-3.3.0/engines';
our $ENGINESDIR_REL     = 'engines';
our $MODULESDIR         = '/mnt/c/Users/user1/Desktop/src/openssl-3.3.0/providers';
our $MODULESDIR_REL     = 'providers';
our $PKGCONFIGDIR       = '';
our $PKGCONFIGDIR_REL   = '';
our $CMAKECONFIGDIR     = '';
our $CMAKECONFIGDIR_REL = '';
our $VERSION            = '3.3.0';
our @LDLIBS             =
    # Unix and Windows use space separation, VMS uses comma separation
    split(/ +| *, */, '-ldl -pthread ');

1;
