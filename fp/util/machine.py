'''
distinguish
'''

import uuid
import socket


def mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[e:e + 2] for e in range(0, 11, 2)])


def host_name():
    return socket.gethostname()


servers = dict(S11=dict(mac='80:18:44:e4:24:b0', hn='R730'),
               S60=dict(mac='ac:1f:6b:24:3f:c2', hn='localhost.localdomain'),
               DGX=dict(mac='75:6a:20:b8:2d:dc', hn='DGX-Station-01'))  # DGX has changed


def is_(machine_name, method='hostname'):
    '''
    Args
        machine_name [str]: name of server, can be
            - 's11': *.*.*.11 server
            - 's60': *.*.*.60 server
            - 'dgx': DGX station
            Both lower-case and upper-case letters are ok.
    '''
    machine_name = machine_name.upper()
    if not machine_name in servers.keys():
        return False
    if method == 'mac_address':
        return mac == servers[machine_name]['mac'] == mac_address()
    elif method == 'hostname':
        return servers[machine_name]['hn'] == host_name()
