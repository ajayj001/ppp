package rubiks.ipl;

import ibis.ipl.Ibis;
import ibis.ipl.IbisCapabilities;
import ibis.ipl.IbisCreationFailedException;
import ibis.ipl.IbisFactory;
import ibis.ipl.IbisIdentifier;
import ibis.ipl.PortType;
import java.io.IOException;

/**
 * Solver for Rubik's cube puzzle.
 * 
 * @author Niels Drost, Timo van Kessel
 * 
 */
public class Rubiks {
    
	/**
     * Port type used for sending a request to the server
     */
    private final static PortType requestPortType = new PortType(
			PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_OBJECT,
			PortType.RECEIVE_AUTO_UPCALLS, PortType.CONNECTION_MANY_TO_ONE);

    /**
     * Port type used for sending a reply back
     */
    private final static PortType replyPortType = new PortType(
			PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_DATA,
			PortType.RECEIVE_EXPLICIT, PortType.CONNECTION_ONE_TO_ONE);

    private final static IbisCapabilities ibisCapabilities =
			new IbisCapabilities(IbisCapabilities.ELECTIONS_STRICT);

    private static Ibis myIbis;
	
	static void run(String[] arguments) throws IbisCreationFailedException, IOException {
		// Create an ibis instance.
        // Notice createIbis uses varargs for its parameters.
        myIbis = IbisFactory.createIbis(ibisCapabilities, null,
                requestPortType, replyPortType);

        // Elect a server
        IbisIdentifier master = myIbis.registry().elect("Master");

        // If I am the server, run server, else run client.
        if (master.equals(myIbis.identifier())) {
            new Master().run(arguments);
        } else {
            new Worker().run(master);
        }

        // End ibis.
        myIbis.end();
	}
	
	public static void main(String[] arguments) {
		try {
            run(arguments);
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
	}
}
