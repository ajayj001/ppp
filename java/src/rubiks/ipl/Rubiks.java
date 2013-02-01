package rubiks.ipl;

import ibis.ipl.Ibis;
import ibis.ipl.IbisCapabilities;
import ibis.ipl.IbisCreationFailedException;
import ibis.ipl.IbisFactory;
import ibis.ipl.IbisIdentifier;
import ibis.ipl.PortType;

/**
 * Solver for Rubik's cube puzzle
 */
public class Rubiks {

	public final static int INIT_VALUE = -1;
	
	public final static PortType portType = new PortType(
			PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_OBJECT_IBIS,
			PortType.RECEIVE_AUTO_UPCALLS, PortType.CONNECTION_MANY_TO_ONE,
			PortType.CONNECTION_UPCALLS);

	public final static IbisCapabilities ibisCapabilities = new IbisCapabilities(
			IbisCapabilities.ELECTIONS_STRICT, IbisCapabilities.CLOSED_WORLD);

	public final Ibis ibis;

	Rubiks() throws IbisCreationFailedException {
		ibis = IbisFactory.createIbis(ibisCapabilities, null,
				portType, portType);
	}

	void run(String[] arguments) throws Exception {
		// Elect a master
		IbisIdentifier master = ibis.registry().elect("Master");

		// If I am the master, run master, else run worker
		if (master.equals(ibis.identifier())) {
			new Master(this).run(arguments);
		} else {
			new Worker(this).run(master);
		}

		// End ibis
		ibis.end();
	}

	public static void main(String[] arguments) {
		try {
			new Rubiks().run(arguments);
		} catch (Exception e) {
			e.printStackTrace(System.err);
		}
	}
}