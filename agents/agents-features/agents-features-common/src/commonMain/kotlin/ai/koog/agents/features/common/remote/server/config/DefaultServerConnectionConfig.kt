package ai.koog.agents.features.common.remote.server.config

/**
 * Default implementation of the server connection configuration.
 *
 * This class provides configuration settings for setting up a server connection,
 * extending the `ServerConnectionConfig` base class. It initializes the server
 * port configuration to a default value unless explicitly specified.
 *
 * @param port The port number on which the server will listen to. Defaults to 8080.
 */
public class DefaultServerConnectionConfig(
    host: String = DEFAULT_HOST,
    port: Int = DEFAULT_PORT,
    wait: Boolean = DEFAULT_WAIT,
) : ServerConnectionConfig(host, port, wait) {

    /**
     * Contains default configurations for server connection parameters.
     *
     * This companion object provides constant values used as default
     * configurations for setting up a server connection. These defaults
     * include the server port, host address, and a suspend flag.
     *
     * These constants are used in the `DefaultServerConnectionConfig` class
     * to provide an initial configuration unless explicitly overridden.
     *
     * @property DEFAULT_PORT The default port number the server will listen to.
     * @property DEFAULT_HOST The default host address for the connection.
     * @property DEFAULT_WAIT Indicates whether the server connection is suspended by default.
     */
    public companion object {

        /**
         * The default port number the server will listen to. Defaults to 50881.
         */
        public const val DEFAULT_PORT: Int = 50881

        internal const val DEFAULT_HOST = "127.0.0.1"

        internal const val DEFAULT_WAIT = false
    }
}
